// Include the main graph header file, which likely contains declarations for graph-related functions and classes
#include <graph/base.cuh>
#include <graph/graph.cuh>
#include <utils/utils.cuh>

// Define a namespace for graph-related operations
namespace graph_{

    // Function to perform unique aggregation. It takes cluster mapping and features as input.
    // Returns a map of string to PyTorch tensors, representing aggregated features for unique clusters.
    std::map<std::string, torch::Tensor> unique_aggregation(torch::Tensor* cluster_map, torch::Tensor* features){
        // Get the number of nodes, intermediate nodes, and features from the input tensors
        const unsigned int n_nodes = cluster_map -> size(0);
        const unsigned int ij_node = cluster_map -> size(1); 
        const unsigned int n_feat  = features -> size(1); 

        // Convert cluster_map to long type and create tensors for unique clusters and output
        torch::Tensor clust  = cluster_map -> to(torch::kLong); 
        torch::Tensor uniq   = cluster_map -> to(torch::kLong); 
        torch::Tensor output = torch::zeros({n_nodes, n_feat}, MakeOp(features)); 

        // Define thread and block dimensions for CUDA kernels
        const dim3 ths = dim3(16, 8, 8); 
        const dim3 bls = blk_(n_nodes, 16, ij_node, 8, ij_node, 8); 

        const dim3 thx = dim3(16, 16); 
        const dim3 blx = blk_(n_nodes, 16, n_feat, 16); 

        // Dispatch CUDA kernels for unique aggregation
        AT_DISPATCH_ALL_TYPES(features -> scalar_type(), "unique_sum", [&]{ 
            _fast_unique<scalar_t><<<bls, ths>>>(
                       uniq.packed_accessor64<long, 2, torch::RestrictPtrTraits>(),
                      clust.packed_accessor64<long, 2, torch::RestrictPtrTraits>(), 
                    n_nodes, ij_node, ij_node);

            _unique_sum<scalar_t><<<blx, thx>>>(
                     output.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                       uniq.packed_accessor64<long    , 2, torch::RestrictPtrTraits>(), 
                features -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    n_nodes, ij_node, n_feat);
        }); 

        // Create a map to store the output tensors
        std::map<std::string, torch::Tensor> out; 
        out["node-sum"] = output; 
        out["unique"] = uniq; 
        return out; 
    }

    // Function to perform edge aggregation. It takes edge indices, prediction, and node features as input.
    // Returns a map of string to PyTorch tensors, representing aggregated edge features.
    std::map<std::string, torch::Tensor> edge_aggregation(torch::Tensor* edge_index, torch::Tensor* prediction, torch::Tensor* node_feature){
        // Get the dimensions of node features and prediction tensors
        const unsigned int node_lx = node_feature -> size({0}); 
        const unsigned int node_fx = node_feature -> size({1}); 
        const unsigned int pred_lx = prediction -> size({-1}); 

        // Prepare edge index tensor for processing
        torch::Tensor _edge_index; 
        const unsigned int dx = edge_index -> size({0});
        const unsigned int dy = edge_index -> size({1}); 
        if (dx == 2 && dy != 2){_edge_index = *edge_index;}
        else {_edge_index = torch::transpose(*edge_index, 0, 1);}
        const unsigned int idx = (dx == 2)*dy + (dx != 2)*dx; 

        // Initialize tensors for edge aggregation
        std::vector<long> dims = {pred_lx, node_lx, node_lx}; 
        torch::Tensor _pred = std::get<1>(prediction -> max({-1})); 
        torch::Tensor _pair = -1*torch::ones(dims, MakeOp(edge_index)); 
        torch::Tensor _pmui = torch::zeros({pred_lx, node_lx, node_fx}, MakeOp(node_feature)); 

        // Define thread and block dimensions for CUDA kernels
        const dim3 thx = dim3(256); 
        const dim3 blx = blk_(idx, 256); 

        const dim3 ths = dim3(32, 8, 4); 
        const dim3 bls = blk_(node_lx, 32, node_fx, 8, pred_lx, 4); 

        // Dispatch CUDA kernels for edge aggregation
        AT_DISPATCH_ALL_TYPES(node_feature -> scalar_type(), "PredictionTopology", [&]{
            _prediction_topology<scalar_t><<<blx, thx>>>(
                     _pair.packed_accessor64<long, 3, torch::RestrictPtrTraits>(), 
               _edge_index.packed_accessor64<long, 2, torch::RestrictPtrTraits>(),
                     _pred.packed_accessor64<long, 1, torch::RestrictPtrTraits>(),
                     idx);

            _edge_summing<scalar_t><<<bls, ths>>>(
                     _pair.packed_accessor64<long    , 3, torch::RestrictPtrTraits>(), 
                     _pmui.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
           node_feature -> packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                     pred_lx, node_lx, node_fx);
        }); 

        // Sort the pair tensor and prepare the output map
        _pair = std::get<0>(_pair.sort(-1, true));
        std::map<std::string, torch::Tensor> out; 
        for (size_t x(0); x < pred_lx; ++x){
            std::string name = "cls::" + std::to_string(x) + "::";
            torch::Tensor mx = std::get<0>((_pair.index({int(x)}) > -1).sum({-1}).max({-1})); 
            out[name + "node-indices"] = _pair.index({int(x), 
                    torch::indexing::Slice(), 
                    torch::indexing::Slice(torch::indexing::None, mx.item<int>())
            }); 
            out[name + "node-sum"] = _pmui.index({int(x)}); 
        }
        return out; 
    }

    // Function to perform node aggregation. It takes edge indices, prediction, and node features as input.
    // Returns a map of string to PyTorch tensors, representing aggregated node features.
    std::map<std::string, torch::Tensor> node_aggregation(torch::Tensor* edge_index, torch::Tensor* prediction, torch::Tensor* node_feature){
        // Get the dimensions of edge indices
        const unsigned int x = edge_index -> size(0); 
        const unsigned int j = edge_index -> size(1); 

        // Prepare edge index tensor for processing
        torch::Tensor e_i; 
        if (x == 2 && j != 2){e_i = *edge_index;}
        else {e_i = torch::transpose(*edge_index, 0, 1);}
        e_i = e_i.index({0, torch::indexing::Slice()});

        // Extract predictions for the edges
        torch::Tensor pred = prediction -> index({e_i}); 

        // Call edge aggregation function and return the result
        return graph_::edge_aggregation(edge_index, &pred, node_feature); 
    }
}


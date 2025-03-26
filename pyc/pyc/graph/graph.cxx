#include <graph/graph.h>
#include <utils/utils.h>

std::map<std::string, torch::Tensor> graph_::edge_aggregation(
        torch::Tensor* edge_index, torch::Tensor* prediction, torch::Tensor* node_feature
){
    const unsigned int max = prediction -> size(1); 
    const unsigned int dim_i = node_feature -> size(0);   
    const unsigned int dim_j = node_feature -> size(1); 

    const unsigned int src_dst = edge_index -> size(0); 
    const unsigned int len = edge_index -> size(1); 
    
    torch::Tensor _edge_index; 
    if (src_dst == 2 && len != 2){_edge_index = *edge_index;}
    else {_edge_index = torch::transpose(*edge_index, 0, 1);}

    torch::Tensor _prediction = std::get<1>(prediction -> max({-1})); 
    const torch::TensorOptions op = MakeOp(prediction); 
    torch::Tensor e_i    = _edge_index.index({0, torch::indexing::Slice()}); 
    torch::Tensor e_j    = _edge_index.index({1, torch::indexing::Slice()}); 
    torch::Tensor pair_m = -torch::ones({dim_i, dim_i}, op.dtype(torch::kLong)); 
    torch::Tensor msk_self = e_i != e_j; 
    torch::Tensor pred = _prediction.view({-1}); 
    torch::Tensor pmu  = torch::zeros({dim_i, dim_j}, op); 
     
    // Add self loops to graph 
    pair_m.index_put_({e_i, e_i}, e_i);
    std::map<std::string, torch::Tensor> output; 
    for (signed int i(0); i < max; ++i){
        std::string name = "cls::" + std::to_string(i) + "::"; 

        // Filter all predictions based on classification
        torch::Tensor msk = (pred == i)*msk_self;  

        // remove edges that do not meet the classification 
        torch::Tensor e_i_p = e_i.index({msk}); 
        torch::Tensor e_j_p = e_j.index({msk});
       
        // record the matrix associated with the classification 
        torch::Tensor pair_m_p = pair_m.clone(); 
        pair_m_p.index_put_({e_i_p, e_j_p}, e_j_p); 
        pair_m_p = std::get<0>(pair_m_p.sort(-1, true)); 

        // aggregate edge nodes
        torch::Tensor pmu_i = pmu.clone();  
        for (signed int k(0); k < dim_i; ++k){
            torch::Tensor this_pair, node; 
            this_pair = pair_m_p.index({k}); 
            this_pair = this_pair.index({this_pair > -1}); 
            node = node_feature -> index({this_pair, torch::indexing::Slice()}).sum(0);  
            pmu_i.index_put_({k}, node); 
        }

        torch::Tensor mx = std::get<0>((pair_m_p > -1).sum(-1).max(-1)); 
        output[name + "node-indices"] = pair_m_p.index({
                torch::indexing::Slice(), 
                torch::indexing::Slice(torch::indexing::None, mx.item<int>())
        }); 
        output[name + "node-sum"] = pmu_i; 
    }
    return output; 
}

std::map<std::string, torch::Tensor> graph_::node_aggregation(
        torch::Tensor* edge_index, torch::Tensor* prediction, torch::Tensor* node_feature
){
    const unsigned int src_dst = edge_index -> size(0); 
    const unsigned int len = edge_index -> size(1); 
    
    torch::Tensor _edge_index; 
    if (src_dst == 2 && len != 2){_edge_index = *edge_index;}
    else {_edge_index = torch::transpose(*edge_index, 0, 1);}
    torch::Tensor e_i = _edge_index.index({0, torch::indexing::Slice()}); 
    torch::Tensor pred = prediction -> index({e_i}).clone(); 
    return graph_::edge_aggregation(edge_index, &pred, node_feature);  
}

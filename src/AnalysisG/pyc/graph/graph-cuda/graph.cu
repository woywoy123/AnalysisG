#include <transform/cartesian-cuda.h>
#include <transform/polar-cuda.h>
#include <physics/physics-cuda.h>
#include <stdio.h>
#include <vector>
#include <map>

#include <c10/cuda/CUDAFunctions.h>
#include <cuda_runtime.h>
#include <cuda.h>

#ifndef GRAPH_CUDA_KERNEL_H
#define GRAPH_CUDA_KERNEL_H
#include "graph-kernel.cu"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), "#x must be on CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "#x must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const dim3 GRBLOCKS(const unsigned int threads, const unsigned int len){
    const dim3 blocks( (len + threads -1) / threads ); 
    return blocks; 
}

const dim3 GRBLOCKS(const unsigned int threads, const unsigned int len, const unsigned int dy){
    const dim3 blocks( (len + threads -1) / threads, dy); 
    return blocks;
}

const dim3 GRBLOCKS(
    const unsigned int threads, const unsigned int len, 
    const unsigned int dy, const unsigned int dz)
{
    const dim3 blocks( (len + threads -1) / threads, dy, dz); 
    return blocks; 
}

static const torch::TensorOptions _MakeOpTen(torch::Tensor v1){
    return torch::TensorOptions().dtype(v1.scalar_type()).device(v1.device()); 
}

std::map<std::string, std::vector<torch::Tensor>> _edge_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature
){
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(node_feature.get_device()); 

    std::map<std::string, std::vector<torch::Tensor>> output; 

    torch::Tensor nf = node_feature.to(torch::kDouble); 
    const torch::TensorOptions op = _MakeOpTen(edge_index); 
    const unsigned int max = prediction.size(1); 
    const unsigned int dim_i = node_feature.size(0); 
    const unsigned int dim_j = node_feature.size(1); 
    const unsigned int threads = 1024; 
    std::vector<long> dims = {max, dim_i, dim_i}; 

    torch::Tensor _edge_index; 
    torch::Tensor pred = std::get<1>(torch::max(prediction, -1)).view({-1}).clone(); 
    const unsigned int pred_l = pred.size(0); 
    const unsigned int x = edge_index.size(0); 
    const unsigned int j = edge_index.size(1); 
    if (x != 2 && j == 2){_edge_index = torch::transpose(edge_index, 0, 1).clone(); }
    else { _edge_index = edge_index.clone(); }

    CHECK_INPUT(pred); 
    CHECK_INPUT(node_feature); 
    torch::Tensor pair_m = torch::ones(dims, op)*-1;  
    const dim3 blk = GRBLOCKS(threads, pred_l, max); 
    AT_DISPATCH_ALL_TYPES(pair_m.scalar_type(), "PredictionTopology", ([&]
    {
        _PredTopo<scalar_t><<< blk, threads >>>(
                pair_m.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), 
           _edge_index.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                  pred.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),  
                pred_l, max);
    })); 
    pair_m = std::get<0>(pair_m.sort(-1, true)); 
    torch::Tensor pmu_i = torch::zeros({max, dim_i, dim_j}, _MakeOpTen(nf));

    const dim3 blk_ = GRBLOCKS(threads, dim_i, dim_j, max); 
    AT_DISPATCH_ALL_TYPES(pair_m.scalar_type(), "EdgeSummation", ([&]{
        _EdgeSummation<scalar_t><<< blk_, threads >>>(
                 pmu_i.packed_accessor64<double, 3, torch::RestrictPtrTraits>(),
                pair_m.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    nf.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),  
                dim_i, dim_j, max);
    })); 

    for (int i(0); i < max; ++i){
        std::stringstream x; 
        std::string name; 
        x << i; 
        x >> name; 
    
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> id2; 
        id2 = torch::unique_dim(pair_m.index({i}), 0, false, true, false); 
        torch::Tensor clusters = std::get<0>(id2); 
        torch::Tensor revert   = std::get<1>(id2);

        unsigned int dim_ = clusters.size(0); 
        torch::Tensor pmu_u = torch::zeros({1, dim_, dim_j}, _MakeOpTen(nf));
        clusters = clusters.view({1, dim_, dim_i}); 
        
        const dim3 blk__ = GRBLOCKS(threads, dim_, dim_j, 1); 
        AT_DISPATCH_ALL_TYPES(clusters.scalar_type(), "EdgeSummation", ([&]
        {
            _EdgeSummation<scalar_t><<< blk__, threads >>>(
                     pmu_u.packed_accessor64<double, 3, torch::RestrictPtrTraits>(),
                  clusters.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                        nf.packed_accessor64<double, 2, torch::RestrictPtrTraits>(),  
                    dim_, dim_j, max);
        })); 
        torch::Tensor clip = std::get<0>((clusters.index({0}) > -1).sum({-1}).max({-1})); 

        output[name] = {
            clusters.index({0, torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, clip.item<int>())}), 
            pmu_u.index({0}).to(torch::kFloat), revert, pmu_i.index({i}).to(torch::kFloat)
        };  
    }
    
    c10::cuda::set_device(current_device);
    return output; 
}


std::map<std::string, std::vector<torch::Tensor>> _node_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction, 
                torch::Tensor node_feature)
{
    torch::Tensor pred, e_i; 
    const unsigned int x = edge_index.size(0); 
    const unsigned int j = edge_index.size(1); 
    if (x != 2 && j == 2){e_i = torch::transpose(edge_index, 0, 1);}
    else {e_i = edge_index;}

    e_i = e_i.index({0, torch::indexing::Slice()});
    pred = prediction.index({e_i}).clone(); 
    return _edge_aggregation(edge_index, pred, node_feature); 
}


std::tuple<torch::Tensor, torch::Tensor> _unique_aggregation(
                torch::Tensor cluster_map, torch::Tensor features)
{
    const auto current_device = c10::cuda::current_device();
    c10::cuda::set_device(cluster_map.get_device()); 

    CHECK_INPUT(cluster_map); 
    CHECK_INPUT(features); 
    const unsigned int n_nodes = cluster_map.size(0);
    const unsigned int ij_node = cluster_map.size(1); 
    const unsigned int n_feat  = features.size(1); 
    const unsigned int threads = 1024; 

    const torch::TensorOptions op  = _MakeOpTen(features);
    const torch::TensorOptions op_ = _MakeOpTen(cluster_map.to(torch::kLong)); 

    torch::Tensor clust  = cluster_map.to(op_).clone(); 
    torch::Tensor uniq   = cluster_map.to(op_).clone(); 
    torch::Tensor output = torch::zeros({n_nodes, n_feat}, op); 

    const dim3 blk  = GRBLOCKS(threads, n_nodes, n_feat); 
    const dim3 blk_ = GRBLOCKS(threads, n_nodes, ij_node, ij_node); 
    AT_DISPATCH_ALL_TYPES(features.scalar_type(), "unique_sum", ([&]
    { 
        _fast_unique<scalar_t><<< blk_, threads >>>(
                   uniq.packed_accessor64<long, 2, torch::RestrictPtrTraits>(),
                  clust.packed_accessor64<long, 2, torch::RestrictPtrTraits>(), 
                n_nodes, ij_node, ij_node);

        _unique_sum<scalar_t><<< blk, threads >>>(
                 output.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                   uniq.packed_accessor64<long, 2, torch::RestrictPtrTraits>(), 
               features.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                n_nodes, ij_node, n_feat);
    })); 

    c10::cuda::set_device(current_device);
    return {output, uniq}; 
}





// --------------------- Interfaces ------------------------ //
std::map<std::string, std::vector<torch::Tensor>> _polar_edge_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu)
{
    torch::Tensor pmc = transform::cuda::PxPyPzE(pmu.to(torch::kFloat64)); 
    return _edge_aggregation(edge_index, prediction, pmc); 
}


std::map<std::string, std::vector<torch::Tensor>> _polar_edge_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction,
                torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e)
{
    std::vector<signed long> d = {-1, 1};
    std::vector<torch::Tensor> pmu = {pt.view(d), eta.view(d), phi.view(d), e.view(d)}; 
    torch::Tensor pmc = transform::cuda::PxPyPzE(torch::cat(pmu, -1).to(torch::kFloat64)); 
    return _edge_aggregation(edge_index, prediction, pmc); 
}

std::map<std::string, std::vector<torch::Tensor>> _polar_node_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu)
{
    torch::Tensor pmc = transform::cuda::PxPyPzE(pmu.to(torch::kFloat64)); 
    return _node_aggregation(edge_index, prediction, pmc); 
}


std::map<std::string, std::vector<torch::Tensor>> _polar_node_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction,
                torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e)
{
    std::vector<signed long> d = {-1, 1};
    std::vector<torch::Tensor> pmu = {pt.view(d), eta.view(d), phi.view(d), e.view(d)}; 
    torch::Tensor pmc = transform::cuda::PxPyPzE(torch::cat(pmu, -1).to(torch::kFloat64)); 
    return _node_aggregation(edge_index, prediction, pmc); 
}

std::map<std::string, std::vector<torch::Tensor>> _cartesian_edge_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc)
{
    return _edge_aggregation(edge_index, prediction, pmc); 
}


std::map<std::string, std::vector<torch::Tensor>> _cartesian_edge_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction,
                torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e)
{
    std::vector<signed long> d = {-1, 1};
    std::vector<torch::Tensor> pmc = {px.view(d), py.view(d), pz.view(d), e.view(d)}; 
    return _edge_aggregation(edge_index, prediction, torch::cat(pmc, -1)); 
}

std::map<std::string, std::vector<torch::Tensor>> _cartesian_node_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc)
{
    return _node_aggregation(edge_index, prediction, pmc); 
}

std::map<std::string, std::vector<torch::Tensor>> _cartesian_node_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction,
                torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e)
{
    std::vector<signed long> d = {-1, 1};
    std::vector<torch::Tensor> pmc = {px.view(d), py.view(d), pz.view(d), e.view(d)}; 
    return _node_aggregation(edge_index, prediction, torch::cat(pmc, -1)); 
}

#endif

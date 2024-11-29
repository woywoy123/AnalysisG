#ifndef CUNUSOL_CUDA_H
#define CUNUSOL_CUDA_H

#include <map>
#include <string>
#include <torch/torch.h>

namespace graph_ {
    std::tuple<torch::Tensor, torch::Tensor> _unique_aggregation(torch::Tensor cluster_map, torch::Tensor features);
    std::map<std::string, torch::Tensor> _polar_edge_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu); 
    std::map<std::string, torch::Tensor> _polar_node_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu); 
    
    std::map<std::string, torch::Tensor> _edge_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature); 
    std::map<std::string, torch::Tensor> _node_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature); 
    
    std::map<std::string, torch::Tensor> _cartesian_edge_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc); 
    std::map<std::string, torch::Tensor> _cartesian_node_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc); 
}

#endif

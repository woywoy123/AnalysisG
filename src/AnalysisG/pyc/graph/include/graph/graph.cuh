#ifndef CUGRAPH_H
#define CUGRAPH_H

#include <map>
#include <string>
#include <torch/torch.h>

namespace graph_ {
    std::map<std::string, torch::Tensor> unique_aggregation(torch::Tensor* cluster_map, torch::Tensor* features);
    std::map<std::string, torch::Tensor> edge_aggregation(torch::Tensor* edge_index, torch::Tensor* prediction, torch::Tensor* node_feature); 
    std::map<std::string, torch::Tensor> node_aggregation(torch::Tensor* edge_index, torch::Tensor* prediction, torch::Tensor* node_feature); 
    std::map<std::string, torch::Tensor> next_selection(torch::Tensor* event_idx, torch::Tensor* batch_idx, torch::Tensor* node_idx, torch::Tensor* edge_idx, long max_node); 
    std::map<std::string, torch::Tensor> cycle_aggregation(torch::Tensor* cluster_map, torch::Tensor* node_feature); 
}

#endif

#ifndef GRAPH_H
#define GRAPH_H

#include <map>
#include <string>
#include <torch/torch.h>

namespace graph_ {
    std::map<std::string, torch::Tensor> unique_aggregation(torch::Tensor* cluster_map, torch::Tensor* features);
    std::map<std::string, torch::Tensor> edge_aggregation(torch::Tensor* edge_index, torch::Tensor* prediction, torch::Tensor* node_feature); 
    std::map<std::string, torch::Tensor> node_aggregation(torch::Tensor* edge_index, torch::Tensor* prediction, torch::Tensor* node_feature); 

}

#endif

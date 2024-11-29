#include <graph/graph.cuh>
#include <cutils/utils.cuh>
#include <physics/physics.cuh>
#include <transform/transform.cuh>
#include <operators/operators.cuh>

std::tuple<torch::Tensor, torch::Tensor> _unique_aggregation(torch::Tensor cluster_map, torch::Tensor features){
    return {}; 
}

std::map<std::string, torch::Tensor> _polar_edge_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu){
    return {}; 
}

std::map<std::string, torch::Tensor> _polar_node_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu){
    return {}; 
}


std::map<std::string, torch::Tensor> _edge_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature){
    return {}; 
}

std::map<std::string, torch::Tensor> _node_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature){
    return {}; 
}


std::map<std::string, torch::Tensor> _cartesian_edge_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc){
    return {}; 
}

std::map<std::string, torch::Tensor> _cartesian_node_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc){
    return {}; 
}




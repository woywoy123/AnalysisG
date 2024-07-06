#include <graph/graph-cuda.h>

std::map<std::string, std::vector<torch::Tensor>> graph::cuda::edge_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature){
    return _edge_aggregation(edge_index, prediction, node_feature); 
}

std::map<std::string, std::vector<torch::Tensor>> graph::cuda::node_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature){
    return _node_aggregation(edge_index, prediction, node_feature); 
}

std::tuple<torch::Tensor, torch::Tensor> graph::cuda::unique_aggregation(torch::Tensor cluster_map, torch::Tensor features){
    return _unique_aggregation(cluster_map, features); 
}


std::map<std::string, std::vector<torch::Tensor>> graph::cuda::polar::edge_pmu(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu){
    return _polar_edge_aggregation(edge_index, prediction, pmu); 
}

std::map<std::string, std::vector<torch::Tensor>> graph::cuda::polar::edge_pmu(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    return _polar_edge_aggregation(edge_index, prediction, pt, eta, phi, e); 
}

std::map<std::string, std::vector<torch::Tensor>> graph::cuda::polar::node_pmu(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu){
    return _polar_node_aggregation(edge_index, prediction, pmu); 
}

std::map<std::string, std::vector<torch::Tensor>> graph::cuda::polar::node_pmu(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
    return _polar_node_aggregation(edge_index, prediction, pt, eta, phi, e); 
}

std::map<std::string, std::vector<torch::Tensor>> graph::cuda::cartesian::edge_pmc(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc){
    return _cartesian_edge_aggregation(edge_index, prediction, pmc); 
}

std::map<std::string, std::vector<torch::Tensor>> graph::cuda::cartesian::edge_pmc(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    return _cartesian_edge_aggregation(edge_index, prediction, px, py, pz, e); 
}

std::map<std::string, std::vector<torch::Tensor>> graph::cuda::cartesian::node_pmc(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc){
    return _cartesian_node_aggregation(edge_index, prediction, pmc); 
}

std::map<std::string, std::vector<torch::Tensor>> graph::cuda::cartesian::node_pmc(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
    return _cartesian_node_aggregation(edge_index, prediction, px, py, pz, e); 
}





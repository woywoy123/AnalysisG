#include <pyc/pyc.h>

#ifdef PYC_CUDA
#include <utils/utils.cuh>
#include <transform/transform.cuh>
#include <graph/graph.cuh>
#else 
#include <utils/utils.h>
#include <transform/transform.h>
#include <graph/graph.h>
#endif

torch::Dict<std::string, torch::Tensor> pyc::graph::edge_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor edge_feature
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    out = graph_::edge_aggregation(&edge_index, &prediction, &edge_feature); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::node_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    out = graph_::node_aggregation(&edge_index, &prediction, &node_feature); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::unique_aggregation(
        torch::Tensor cluster_map, torch::Tensor feature
){
    changedev(&cluster_map); 
    std::map<std::string, torch::Tensor> out; 
    out = graph_::unique_aggregation(&cluster_map, &feature); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::polar::edge_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    out = graph_::edge_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::polar::node_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    torch::Tensor pmc = transform_::PxPyPzE(&pmu); 
    out = graph_::edge_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::polar::edge_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, 
        torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    out = graph_::edge_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::polar::node_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, 
        torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    torch::Tensor pmc = transform_::PxPyPzE(&pt, &eta, &phi, &e); 
    out = graph_::node_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::cartesian::edge_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    out = graph_::edge_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::cartesian::node_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    out = graph_::node_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::cartesian::edge_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, 
        torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    torch::Tensor pmc = torch::cat({px, py, pz, e}, {-1}); 
    out = graph_::edge_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}

torch::Dict<std::string, torch::Tensor> pyc::graph::cartesian::node_aggregation(
        torch::Tensor edge_index, torch::Tensor prediction, 
        torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e
){
    changedev(&edge_index); 
    std::map<std::string, torch::Tensor> out; 
    torch::Tensor pmc = torch::cat({px, py, pz, e}, {-1}); 
    out = graph_::node_aggregation(&edge_index, &prediction, &pmc); 
    return pyc::std_to_dict(&out); 
}


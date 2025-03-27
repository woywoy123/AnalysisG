#include <pyc/pyc.h>

#ifdef PYC_CUDA
#include <utils/utils.cuh>
#include <transform/transform.cuh>
#include <graph/graph.cuh>
#include <graph/pagerank.cuh>
#else 
#include <utils/utils.h>
#include <transform/transform.h>
#include <graph/graph.h>
#endif

torch::Dict<std::string, torch::Tensor> pyc::graph::PageRank(
        torch::Tensor edge_index, torch::Tensor edge_scores, 
        double alpha, double threshold, double norm_low, long timeout, long num_cls
){
    changedev(&edge_index); changedev(&edge_scores); 
    const unsigned int edx = edge_index.size({0});
    const unsigned int edy = edge_index.size({1}); 
    if (edx != 2 && edy == 2){edge_index = edge_index.transpose(0, 1).contiguous();}

    const unsigned int sdx = edge_scores.size({0});
    const unsigned int sdy = edge_scores.size({1}); 
    if (sdx != 2 && sdy == 2){edge_scores = edge_scores.transpose(0, 1).contiguous();}

    std::map<std::string, torch::Tensor> out; 
    #ifdef PYC_CUDA
    out = graph_::page_rank(&edge_index, &edge_scores, alpha, threshold, norm_low, timeout, num_cls);
    #endif
    return pyc::std_to_dict(&out); 
}


torch::Dict<std::string, torch::Tensor> pyc::graph::PageRankReconstruction(
        torch::Tensor edge_index, torch::Tensor edge_scores, torch::Tensor pmc, 
        double alpha, double threshold, double norm_low, long timeout, long num_cls
){

    changedev(&edge_index); changedev(&edge_scores); changedev(&pmc); 
    const unsigned int edx = edge_index.size({0});
    const unsigned int edy = edge_index.size({1}); 
    if (edx != 2 && edy == 2){edge_index = edge_index.transpose(0, 1).contiguous();}

    const unsigned int sdx = edge_scores.size({0});
    const unsigned int sdy = edge_scores.size({1}); 
    if (sdx != 2 && sdy == 2){edge_scores = edge_scores.softmax(-1).transpose(0, 1).contiguous();}
    else {edge_scores = edge_scores.softmax(0);}

    std::map<std::string, torch::Tensor> out; 
    #ifdef PYC_CUDA
    out = graph_::page_rank_reconstruction(&edge_index, &edge_scores, &pmc, alpha, threshold, norm_low, timeout, num_cls);
    #endif

    return pyc::std_to_dict(&out); 
}


















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


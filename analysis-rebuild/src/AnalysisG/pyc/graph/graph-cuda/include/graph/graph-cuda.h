#ifndef H_GRAPH_TENSOR 
#define H_GRAPH_CUDA
#include <torch/torch.h>

std::tuple<torch::Tensor, torch::Tensor> _unique_aggregation(torch::Tensor cluster_map, torch::Tensor features);
std::map<std::string, std::vector<torch::Tensor>> _polar_edge_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu); 
std::map<std::string, std::vector<torch::Tensor>> _polar_node_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu); 

std::map<std::string, std::vector<torch::Tensor>> _edge_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature); 
std::map<std::string, std::vector<torch::Tensor>> _node_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature); 

std::map<std::string, std::vector<torch::Tensor>> _cartesian_edge_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc); 
std::map<std::string, std::vector<torch::Tensor>> _cartesian_node_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc); 

std::map<std::string, std::vector<torch::Tensor>> _polar_edge_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e);
std::map<std::string, std::vector<torch::Tensor>> _polar_node_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e); 

std::map<std::string, std::vector<torch::Tensor>> _cartesian_node_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e);
std::map<std::string, std::vector<torch::Tensor>> _cartesian_edge_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 

namespace graph {
    namespace cuda {
        std::map<std::string, std::vector<torch::Tensor>> edge_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature); 
        std::map<std::string, std::vector<torch::Tensor>> node_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature); 
        std::tuple<torch::Tensor, torch::Tensor> unique_aggregation(torch::Tensor cluster_map, torch::Tensor features); 

        namespace polar {
            std::map<std::string, std::vector<torch::Tensor>> edge_pmu(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu); 
            std::map<std::string, std::vector<torch::Tensor>> edge_pmu(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e);
            std::map<std::string, std::vector<torch::Tensor>> node_pmu(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu);
            std::map<std::string, std::vector<torch::Tensor>> node_pmu(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e); 
        }

        namespace cartesian {
            std::map<std::string, std::vector<torch::Tensor>> edge_pmc(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc); 
            std::map<std::string, std::vector<torch::Tensor>> edge_pmc(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 
            std::map<std::string, std::vector<torch::Tensor>> node_pmc(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc); 
            std::map<std::string, std::vector<torch::Tensor>> node_pmc(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 
        }
    }
}


#endif

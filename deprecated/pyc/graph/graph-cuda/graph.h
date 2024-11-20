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

namespace Graph {
    namespace CUDA {
        inline std::map<std::string, std::vector<torch::Tensor>> edge_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature){
            return _edge_aggregation(edge_index, prediction, node_feature); 
        }

        inline std::map<std::string, std::vector<torch::Tensor>> node_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature){
            return _node_aggregation(edge_index, prediction, node_feature); 
        }

        inline std::tuple<torch::Tensor, torch::Tensor> unique_aggregation(torch::Tensor cluster_map, torch::Tensor features){
            return _unique_aggregation(cluster_map, features); 
        }


        namespace Polar {
            inline std::map<std::string, std::vector<torch::Tensor>> edge_pmu(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu){
                return _polar_edge_aggregation(edge_index, prediction, pmu); 
            }

            inline std::map<std::string, std::vector<torch::Tensor>> edge_pmu(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
                return _polar_edge_aggregation(edge_index, prediction, pt, eta, phi, e); 
            }

            inline std::map<std::string, std::vector<torch::Tensor>> node_pmu(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu){
                return _polar_node_aggregation(edge_index, prediction, pmu); 
            }

            inline std::map<std::string, std::vector<torch::Tensor>> node_pmu(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e){
                return _polar_node_aggregation(edge_index, prediction, pt, eta, phi, e); 
            }
        }

        namespace Cartesian {

            inline std::map<std::string, std::vector<torch::Tensor>> edge_pmc(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc){
                return _cartesian_edge_aggregation(edge_index, prediction, pmc); 
            }

            inline std::map<std::string, std::vector<torch::Tensor>> edge_pmc(
                    torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
                return _cartesian_edge_aggregation(edge_index, prediction, px, py, pz, e); 
            }

            inline std::map<std::string, std::vector<torch::Tensor>> node_pmc(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc){
                return _cartesian_node_aggregation(edge_index, prediction, pmc); 
            }

            inline std::map<std::string, std::vector<torch::Tensor>> node_pmc(
                    torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e){
                return _cartesian_node_aggregation(edge_index, prediction, px, py, pz, e); 
            }
        }
    }
}


#endif

#ifndef H_GRAPH_TENSOR 
#define H_GRAPH_TENSOR 
#include <torch/torch.h>
#include <transform/cartesian-tensors/cartesian.h>
#include <transform/polar-tensors/polar.h>
#include <physics/physics-tensor/physics.h>

namespace Graph {
    namespace Tensors {
        torch::TensorOptions MakeOp(torch::Tensor x);
        std::map<std::string, std::vector<torch::Tensor>> edge_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature); 
        std::map<std::string, std::vector<torch::Tensor>> node_aggregation(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor node_feature); 

        namespace Polar {
            std::map<std::string, std::vector<torch::Tensor>> edge_pmu(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu); 
            std::map<std::string, std::vector<torch::Tensor>> node_pmu(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmu); 
            std::map<std::string, std::vector<torch::Tensor>> edge_pmu(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e); 
            std::map<std::string, std::vector<torch::Tensor>> node_pmu(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pt, torch::Tensor eta, torch::Tensor phi, torch::Tensor e); 
        }

        namespace Cartesian {
             std::map<std::string, std::vector<torch::Tensor>> edge_pmc(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc); 
             std::map<std::string, std::vector<torch::Tensor>> node_pmc(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor pmc); 
             std::map<std::string, std::vector<torch::Tensor>> edge_pmc(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 
             std::map<std::string, std::vector<torch::Tensor>> node_pmc(torch::Tensor edge_index, torch::Tensor prediction, torch::Tensor px, torch::Tensor py, torch::Tensor pz, torch::Tensor e); 
        }
    }
}

#endif

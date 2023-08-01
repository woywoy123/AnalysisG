#ifndef H_GRAPH_TENSOR 
#define H_GRAPH_TENSOR 
#include <torch/torch.h>
#include <transform/cartesian-tensors/cartesian.h>
#include <transform/polar-tensors/polar.h>
#include <physics/physics-tensor/physics.h>


namespace Graph
{
    namespace Tensors
    {
        std::map<std::string, torch::Tensor> edge_aggregation(
                torch::Tensor edge_index, torch::Tensor prediction, 
                torch::Tensor node_feature, const bool include_zero); 

    }

}


#endif

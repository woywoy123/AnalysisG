#include <utils.h>
#include <pyc/pyc.h>

torch::Tensor utils::NRecode(
        torch::Tensor pmc, torch::Tensor num_node, torch::Tensor* node_rnn
){
    torch::Tensor feats = (num_node > -1).sum({-1}, true); 
    torch::Tensor mass  = pyc::physics::cartesian::combined::M(pmc); 
    torch::Tensor nox   = torch::cat({mass, pmc, feats, *node_rnn}, {-1});
    nox = (*this -> rnn_x) -> forward(nox.to(torch::kFloat32)); 
    return nox / num_node.to(torch::kFloat32);  
}


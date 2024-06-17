#include <RecursiveGraphNeuralNetwork.h>
#include <transform/cartesian-cuda.h>
#include <physics/cartesian-cuda.h>
#include <transform/polar-cuda.h>
#include <nusol/nusol-cuda.h>
#include <graph/graph-cuda.h>


recursivegraphneuralnetwork::recursivegraphneuralnetwork(){
    this -> rnn_dx = new torch::nn::Sequential({
            {"rnn_dx_l1", torch::nn::Linear(this -> _dx, this -> _hidden)},
            {"rnn_dx_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_dx_l2", torch::nn::Linear(this -> _hidden, this -> _repeat)}
    }); 
    this -> register_module(this -> rnn_dx); 

    this -> rnn_x = new torch::nn::Sequential({
            {"rnn_x_l1", torch::nn::Linear(this -> _x, this -> _hidden)}, 
            {"rnn_x_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_x_l2", torch::nn::Linear(this -> _hidden, this -> _repeat)}
    });
    this -> register_module(this -> rnn_x); 

    this -> rnn_mrg = new torch::nn::Sequential({
            {"rnn_mrg_l1", torch::nn::Linear(this -> _repeat*2, this -> _hidden)}, 
            {"rnn_mrg_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_mrg_l2", torch::nn::Linear(this -> _hidden, this -> _o)}
    });
    this -> register_module(this -> rnn_mrg); 
}

torch::Tensor recursivegraphneuralnetwork::message(
    torch::Tensor trk_i, torch::Tensor trk_j, 
    torch::Tensor pmc_i, torch::Tensor pmc_j, torch::Tensor pmc
){
    torch::Tensor pmci = std::get<0>(graph::cuda::unique_aggregation(trk_i, pmc)); 
    torch::Tensor pmcj = std::get<0>(graph::cuda::unique_aggregation(trk_j, pmc)); 
    std::tuple<torch::Tensor, torch::Tensor> pmc_ij = graph::cuda::unique_aggregation(torch::cat({trk_i, trk_j}, {-1}), pmc); 
    
    torch::Tensor m_i  = physics::cuda::cartesian::M(pmc_i);
    torch::Tensor m_j  = physics::cuda::cartesian::M(pmc_j); 
    torch::Tensor m_ij = physics::cuda::cartesian::M(std::get<0>(pmc_ij));
    torch::Tensor dR   = physics::cuda::cartesian::DeltaR(pmc_i, pmc_j); 
    torch::Tensor jmp  = (std::get<1>(pmc_ij) > -1).sum({-1}).view({-1, 1}); 
    
    std::vector<torch::Tensor> dx = {m_j, m_j-m_i, pmc_j, pmc_j - pmc_i}; 
    torch::Tensor hdx = (*this -> rnn_dx) -> forward(torch::cat(dx, {-1}).to(torch::kFloat32)); 
   
    std::vector<torch::Tensor> x = {m_ij, dR, jmp, std::get<0>(pmc_ij)}; 
    torch::Tensor hx = (*this -> rnn_x) -> forward(torch::cat(x, {-1}).to(torch::kFloat32)); 
    return (*this -> rnn_mrg) -> forward(torch::cat({hx, hdx}, {-1})); 
}


void recursivegraphneuralnetwork::forward(graph_t* data){

    torch::Tensor edge_index = data -> edge_index -> clone(); 
    torch::Tensor src = edge_index.index({0}); 
    torch::Tensor dst = edge_index.index({1}); 

    torch::Tensor pt      = data -> get_data_node("pt") -> clone();
    torch::Tensor eta     = data -> get_data_node("eta") -> clone(); 
    torch::Tensor phi     = data -> get_data_node("phi") -> clone(); 
    torch::Tensor energy  = data -> get_data_node("energy") -> clone(); 
    torch::Tensor met     = data -> get_data_graph("met") -> clone(); 
    torch::Tensor met_phi = data -> get_data_graph("phi") -> clone(); 
    torch::Tensor met_xy  = torch::cat({transform::cuda::Px(met, met_phi), transform::cuda::Py(met, met_phi)}, {-1}); 

    torch::Tensor pmu = torch::cat({pt, eta, phi, energy}, {-1}); 
    torch::Tensor pmc = transform::cuda::PxPyPzE(pmu); 
    torch::Tensor trk_ = torch::ones_like(pt).cumsum(0) -1; 

    torch::Tensor H = this -> message(trk_.index({src}), trk_.index({dst}), pmu.index({src}), pmu.index({dst}), pmc);  
    this -> prediction_edge_feature("top_edge", H); 
}

recursivegraphneuralnetwork::~recursivegraphneuralnetwork(){}
model_template* recursivegraphneuralnetwork::clone(){
    return new recursivegraphneuralnetwork(); 
}

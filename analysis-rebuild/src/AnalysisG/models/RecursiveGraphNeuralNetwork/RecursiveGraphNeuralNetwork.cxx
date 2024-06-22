#include <RecursiveGraphNeuralNetwork.h>
#include <transform/cartesian-cuda.h>
#include <physics/cartesian-cuda.h>
#include <transform/polar-cuda.h>
#include <nusol/nusol-cuda.h>
#include <graph/graph-cuda.h>


recursivegraphneuralnetwork::recursivegraphneuralnetwork(){

    this -> rnn_mass = new torch::nn::Sequential({
            {"rnn_mass_l1", torch::nn::Linear(this -> _dx, this -> _hidden)},
            {"rnn_mass_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_mass_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)},
    }); 
    this -> register_module(this -> rnn_mass); 

    this -> rnn_mass_dx = new torch::nn::Sequential({
            {"rnn_mass_dx_l1", torch::nn::Linear(this -> _dx*2, this -> _hidden)},
            {"rnn_mass_dx_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_mass_dx_relu", torch::nn::ReLU()},
            {"rnn_mass_dx_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)}
    }); 
    this -> register_module(this -> rnn_mass_dx); 

    this -> rnn_mass_mrg = new torch::nn::Sequential({
            {"rnn_mass_mrg_l1", torch::nn::Linear(this -> _hidden*2, this -> _hidden*2)},
            {"rnn_mass_mrg_relu", torch::nn::ReLU()},
            {"rnn_mass_mrg_l2", torch::nn::Linear(this -> _hidden*2, this -> _repeat)}
    }); 
    this -> register_module(this -> rnn_mass_mrg); 


    this -> rnn_H = new torch::nn::Sequential({
            {"rnn_H_l1", torch::nn::Linear(this -> _repeat*2, this -> _repeat*2)},
            {"rnn_H_relu", torch::nn::ReLU()},
            {"rnn_H_l2", torch::nn::Linear(this -> _repeat*2, this -> _repeat)}
    }); 
    this -> register_module(this -> rnn_H); 
}

torch::Tensor recursivegraphneuralnetwork::message(
    torch::Tensor _trk_i, torch::Tensor _trk_j, torch::Tensor pmc
){

    torch::Tensor trk_ij = torch::cat({_trk_i, _trk_j}, {-1}); 

    torch::Tensor pmc_i  = std::get<0>(graph::cuda::unique_aggregation(_trk_i, pmc)); 
    torch::Tensor pmc_j  = std::get<0>(graph::cuda::unique_aggregation(_trk_j, pmc)); 
    torch::Tensor pmc_ij = std::get<0>(graph::cuda::unique_aggregation(trk_ij, pmc)); 
   
    torch::Tensor m_ij = physics::cuda::cartesian::M(pmc_ij);
    torch::Tensor m_rnn = (*this -> rnn_mass) -> forward(m_ij.to(torch::kFloat32)); 

    torch::Tensor m_i  = physics::cuda::cartesian::M(pmc_i);
    torch::Tensor m_j  = physics::cuda::cartesian::M(pmc_j); 
    torch::Tensor m_dx = torch::cat({m_i, m_i - m_j}, {-1}).to(torch::kFloat32); 
    torch::Tensor m_rnn_dx = (*this -> rnn_mass_dx) -> forward(m_dx); 
    return (*this -> rnn_mass_mrg) -> forward(torch::cat({m_rnn, m_rnn_dx}, {-1})); 
}

torch::Tensor recursivegraphneuralnetwork::aggregation(
        torch::Tensor edge_index, torch::Tensor pmc, torch::Tensor G
){

    torch::Tensor mlp_h = G.softmax(-1); 
    torch::Tensor src_ = edge_index.index({torch::indexing::Slice(), 0}).view({-1}); 
    torch::Tensor dst_ = edge_index.index({torch::indexing::Slice(), 1}).view({-1});

    torch::Tensor gamma_0 = mlp_h.index({torch::indexing::Slice(), 0}); 
    torch::Tensor gamma_1 = mlp_h.index({torch::indexing::Slice(), 1}); 

    torch::Tensor matrix_0 = this -> matrix -> clone();
    matrix_0.index_put_({src_, dst_}, gamma_0); 

    torch::Tensor matrix_1 = this -> matrix -> clone(); 
    matrix_1.index_put_({src_, dst_}, gamma_1); 

    torch::Tensor matrix_bern = matrix_1*(1-matrix_0); 

    torch::Tensor nxt_n  = std::get<1>(matrix_bern.max({-1})).view({-1, 1});
    torch::Tensor trk_ij = torch::cat({*(this -> trk_i), nxt_n}, {-1}); 

    torch::Tensor trk_self = trk_ij.index({torch::indexing::Slice(), 0}); 
    this -> matrix -> index_put_({trk_self, nxt_n.view({-1})}, -1); 

    torch::Tensor msk    = (this -> matrix -> index({src_, dst_})) > -1;  
    torch::Tensor idx_pr = (this -> idx -> index({src_, dst_})).index({msk});
    if (!idx_pr.size({0})){return G;}

    this -> trk_i = &trk_ij; 
    src_ = this -> trk_i -> index({src_.index({msk})}); 
    dst_ = dst_.index({msk}).view({-1, 1}); 

    torch::Tensor H  = this -> message(src_, dst_, pmc); 
    torch::Tensor H_ = this -> aggregation(edge_index.index({msk}), pmc, H); 
    H_ = torch::cat({G.index({msk}), G.index({msk}) - H_}, {-1}); 
    torch::Tensor G_ = (*this -> rnn_H) -> forward(H_); 
    torch::Tensor O_ = G.clone(); 
    O_.index_put_({msk}, H*G_); 
    return O_; 
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
    
    torch::Tensor truth = data -> get_truth_edge("top_edge") -> clone(); 

    torch::Tensor met_xy  = torch::cat({transform::cuda::Px(met, met_phi), transform::cuda::Py(met, met_phi)}, {-1}); 
    torch::Tensor pmc = transform::cuda::PxPyPzE(pt, eta, phi, energy); 

    torch::Tensor src_ = src.view({-1, 1}); 
    torch::Tensor dst_ = dst.view({-1, 1});

    torch::Tensor trk = (torch::ones_like(pt).cumsum({0})-1).to(torch::kLong); 
    this -> trk_i = &trk; 

    torch::Tensor m_idx = torch::zeros({pmc.size({0}), pmc.size({0})}, pmc.device()).to(torch::kLong);
    m_idx.index_put_({trk.index({src}).view({-1}), trk.index({dst}).view({-1})}, torch::ones_like(src).cumsum({-1})-1);  
    this -> idx = &m_idx; 

    torch::Tensor m_matrix = torch::ones({pmc.size({0}), pmc.size({0})}, pmc.device()); 
    this -> matrix = &m_matrix; 
    torch::Tensor H  = this -> message(src_, dst_, pmc);  
    torch::Tensor H_ = this -> aggregation(torch::cat({src_, dst_}, {-1}), pmc, H);
    this -> prediction_edge_feature("top_edge", H_); 
}

recursivegraphneuralnetwork::~recursivegraphneuralnetwork(){}
model_template* recursivegraphneuralnetwork::clone(){
    return new recursivegraphneuralnetwork(); 
}

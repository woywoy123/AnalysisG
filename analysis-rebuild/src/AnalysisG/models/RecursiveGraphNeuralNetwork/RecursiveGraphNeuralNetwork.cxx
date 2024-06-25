#include <RecursiveGraphNeuralNetwork.h>
#include <transform/cartesian-cuda.h>
#include <physics/cartesian-cuda.h>
#include <transform/polar-cuda.h>
#include <nusol/nusol-cuda.h>
#include <graph/graph-cuda.h>


recursivegraphneuralnetwork::recursivegraphneuralnetwork(){

    this -> rnn_dx = new torch::nn::Sequential({
            {"rnn_dx_l1", torch::nn::Linear(this -> _dx*2, this -> _hidden)},
            {"rnn_dx_tanh_1", torch::nn::Tanh()}, 
            {"rnn_dx_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_dx_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)},
            {"rnn_dx_tanh_2", torch::nn::Tanh()}, 
            {"rnn_dx_l3", torch::nn::Linear(this -> _hidden, this -> _dx)}
    }); 
    this -> register_module(this -> rnn_dx); 


    this -> rnn_node = new torch::nn::Sequential({
            {"rnn_node_l1", torch::nn::Linear(this -> _node, this -> _hidden)},
            {"rnn_node_l2", torch::nn::Linear(this -> _hidden, this -> _dx)}
    }); 
    this -> register_module(this -> rnn_node); 

    
    this -> rnn_repeat = new torch::nn::Sequential({
            {"rnn_repeat_l1", torch::nn::Linear(this -> _dx*4, this -> _hidden)},
            {"rnn_repeat_tanh_1", torch::nn::Tanh()},
            {"rnn_repeat_l2", torch::nn::Linear(this -> _hidden, this -> _output)}, 
    }); 
    this -> register_module(this -> rnn_repeat); 
}

torch::Tensor recursivegraphneuralnetwork::message(
    torch::Tensor _trk_i, torch::Tensor _trk_j, torch::Tensor pmc, 
    torch::Tensor H, torch::Tensor H_
){

    torch::Tensor trk_ij = torch::cat({_trk_i, _trk_j}, {-1}); 

    torch::Tensor pmc_i  = std::get<0>(graph::cuda::unique_aggregation(_trk_i, pmc)); 
    torch::Tensor pmc_j  = std::get<0>(graph::cuda::unique_aggregation(_trk_j, pmc)); 
    torch::Tensor pmc_ij = std::get<0>(graph::cuda::unique_aggregation(trk_ij, pmc)); 
   
    torch::Tensor m_ij = physics::cuda::cartesian::M(pmc_ij);
    torch::Tensor m_i  = physics::cuda::cartesian::M(pmc_i);
    torch::Tensor m_j  = physics::cuda::cartesian::M(pmc_j); 
    
    std::vector<torch::Tensor> features = {m_ij, m_i, m_ij - m_i, H - H_}; 
    torch::Tensor m_dx = torch::cat(features, {-1}).to(torch::kFloat32); 
    return (*this -> rnn_dx) -> forward(m_dx); 
}

torch::Tensor recursivegraphneuralnetwork::aggregation(
        torch::Tensor src, torch::Tensor dst, torch::Tensor pmc, 
        torch::Tensor H, torch::Tensor H_
){ 
    torch::Tensor src_ = src.index({torch::indexing::Slice(), 0}).view({-1}); 

    torch::Tensor node_mass = std::get<0>(graph::cuda::unique_aggregation(src, pmc)); 
    node_mass = physics::cuda::cartesian::M(node_mass).to(torch::kFloat32); 

    torch::Tensor H_e = this -> message(src, dst, pmc, H, H_);  
    torch::Tensor H_n = (*this -> rnn_node) -> forward(node_mass); 
    
    torch::Tensor rep = (*this -> rnn_repeat) -> forward(torch::cat({H_n, H_n - H_, H_e, H - H_e}, {-1})); 
    torch::Tensor sel = std::get<1>(rep.max({-1})) > 0; 
    if (!sel.index({sel}).size({0})){return rep;}

    // get the current node state by selecting the highest non zero edge score 
    torch::Tensor rep_smx = rep.softmax(-1).index({torch::indexing::Slice(), 1}); 
    torch::Tensor matrix = torch::zeros_like(*this -> _matrix); 
    matrix.index_put_({src_, dst.view({-1})}, sel*rep_smx); 
    matrix = matrix*(*this -> _matrix); 
    
    torch::Tensor nulls = matrix.sum({-1}) == 0; 
    torch::Tensor valid = matrix.sum({-1}) == 1; 
    if (!valid.index({valid}).size({0})){return rep;}

    torch::Tensor next_node = std::get<1>(matrix.max({-1})); 
    next_node.index_put_({nulls}, -1); 


    // remove selected next node from matrix, provided it was not previously selected
    this -> _matrix -> index_put_({valid, next_node.index({valid})}, 0);
    
    // update step
    src = torch::cat({src, next_node.index({src_}).view({-1, 1})}, {-1}); 
    return this -> aggregation(src, dst, pmc, H_e, H_n);
}

void recursivegraphneuralnetwork::forward(graph_t* data){

    torch::Tensor pt      = data -> get_data_node("pt") -> clone();
    torch::Tensor eta     = data -> get_data_node("eta") -> clone(); 
    torch::Tensor phi     = data -> get_data_node("phi") -> clone(); 
    torch::Tensor energy  = data -> get_data_node("energy") -> clone(); 
    torch::Tensor pmc     = transform::cuda::PxPyPzE(pt, eta, phi, energy); 

    torch::Tensor met     = data -> get_data_graph("met") -> clone(); 
    torch::Tensor met_phi = data -> get_data_graph("phi") -> clone();     
    torch::Tensor met_xy  = torch::cat({transform::cuda::Px(met, met_phi), transform::cuda::Py(met, met_phi)}, {-1});

    torch::Tensor truth = data -> get_truth_edge("top_edge") -> clone(); 
    torch::Tensor src = data -> edge_index -> index({0}).view({-1, 1}); 
    torch::Tensor dst = data -> edge_index -> index({1}).view({-1, 1}); 

    std::vector<torch::Tensor> H0 = {}; 
    for (size_t x(0); x < this -> _dx; ++x){H0.push_back(torch::zeros_like(src));}
    torch::Tensor H = torch::cat(H0, {-1});  
    torch::Tensor matrix = torch::ones({pmc.size({0}), pmc.size({0})}, src.device()); 
    this -> _matrix = &matrix; 
    H = this -> aggregation(src, dst, pmc, H, H);
    this -> _matrix = nullptr; 
    this -> prediction_edge_feature("top_edge", H); 
}

recursivegraphneuralnetwork::~recursivegraphneuralnetwork(){}
model_template* recursivegraphneuralnetwork::clone(){
    return new recursivegraphneuralnetwork(); 
}

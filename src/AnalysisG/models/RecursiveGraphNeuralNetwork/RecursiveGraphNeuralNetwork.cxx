#include <RecursiveGraphNeuralNetwork.h>
#include <transform/cartesian-cuda.h>
#include <physics/cartesian-cuda.h>
#include <transform/polar-cuda.h>
#include <nusol/nusol-cuda.h>
#include <graph/graph-cuda.h>


recursivegraphneuralnetwork::recursivegraphneuralnetwork(){

    this -> rnn_dx = new torch::nn::Sequential({
            {"rnn_dx_l1", torch::nn::Linear(this -> _dx*3 + this -> _hidden, this -> _hidden)},
            {"rnn_dx_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)}
    }); 


    this -> rnn_node = new torch::nn::Sequential({
            {"rnn_node_l1", torch::nn::Linear(this -> _node, this -> _hidden)},
            {"rnn_node_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)}
    }); 


    this -> rnn_repeat = new torch::nn::Sequential({
            {"rnn_repeat_l1", torch::nn::Linear(this -> _hidden*2, this -> _hidden*2)},
            {"rnn_repeat_l2", torch::nn::Linear(this -> _hidden*2, this -> _output)}
    }); 


    this -> rnn_merge = new torch::nn::Sequential({
            {"rnn_merge_l1", torch::nn::Linear(this -> _output*2, this -> _output*2)},
            {"rnn_merge_l2", torch::nn::Linear(this -> _output*2, this -> _output)}
    }); 

    this -> register_module(this -> rnn_dx); 
    this -> register_module(this -> rnn_node); 
    this -> register_module(this -> rnn_repeat); 
    this -> register_module(this -> rnn_merge); 
}

torch::Tensor recursivegraphneuralnetwork::message(
    torch::Tensor _trk_i, torch::Tensor _trk_j, torch::Tensor pmc, 
    torch::Tensor H_i, torch::Tensor H_j
){

    torch::Tensor trk_ij = torch::cat({_trk_i, _trk_j}, {-1}); 

    torch::Tensor pmc_i  = std::get<0>(graph::cuda::unique_aggregation(_trk_i, pmc)); 
    torch::Tensor pmc_j  = std::get<0>(graph::cuda::unique_aggregation(_trk_j, pmc)); 
    torch::Tensor pmc_ij = std::get<0>(graph::cuda::unique_aggregation(trk_ij, pmc)); 
   
    torch::Tensor m_ij = physics::cuda::cartesian::M(pmc_ij);
    torch::Tensor m_i  = physics::cuda::cartesian::M(pmc_i);
    torch::Tensor m_j  = physics::cuda::cartesian::M(pmc_j); 
    
    std::vector<torch::Tensor> features = {m_ij, m_i, m_j, H_j + H_i}; 
    torch::Tensor m_dx = torch::cat(features, {-1}).to(torch::kFloat32); 
    return (*this -> rnn_dx) -> forward(m_dx); 
}

torch::Tensor recursivegraphneuralnetwork::aggregation(
        torch::Tensor src, torch::Tensor dst, torch::Tensor pmc, torch::Tensor H
){ 
    torch::Tensor path = this -> _path -> clone(); 
    torch::Tensor src_ = src.view({-1});
    torch::Tensor dst_ = dst.view({-1}); 

    torch::Tensor node     = std::get<0>(graph::cuda::unique_aggregation(path, pmc)); 
    torch::Tensor mass     = physics::cuda::cartesian::M(node).to(torch::kFloat32); 
    torch::Tensor n_score  = (*this -> rnn_node) -> forward(torch::cat({mass}, {-1})); 
    torch::Tensor e_score  = this -> message(src, dst, pmc, n_score.index({src_}), n_score.index({dst_})); 
    torch::Tensor edge     = (*this -> rnn_repeat) -> forward(torch::cat({e_score, n_score.index({src_}) - H.index({dst_})}, {-1})); 

    torch::Tensor e_sfmx_ = edge.index({torch::indexing::Slice(), 1});
    torch::Tensor e_selx  = std::get<1>(edge.max({-1})); 

    torch::Tensor matrix  = this -> _matrix -> clone(); 
    matrix.index_put_({src_, dst_}, e_sfmx_); 
    matrix = matrix.softmax(-1); 
    matrix.index_put_({src_, dst_}, e_selx.to(torch::kFloat)); 
    matrix = matrix*(*this -> _matrix);


    torch::Tensor valid = matrix.sum({-1}) > 0; 
    if (!valid.index({valid}).size({0})){return edge;}
    torch::Tensor next_node = std::get<1>(matrix.max({-1})); 
    next_node.index_put_({matrix.sum({-1}) == 0}, -1); 

    this -> _matrix -> index_put_({valid, next_node.index({valid})}, 0);
    (*this -> _path) = torch::cat({path, next_node.view({-1, 1})}, {-1}); 

    torch::Tensor _out = this -> aggregation(src, dst, pmc, n_score);
    return (*this -> rnn_merge) -> forward(torch::cat({_out, edge - _out}, {-1})) * edge.softmax(-1); 
}

void recursivegraphneuralnetwork::forward(graph_t* data){

    torch::Tensor pt      = data -> get_data_node("pt") -> clone();
    torch::Tensor eta     = data -> get_data_node("eta") -> clone(); 
    torch::Tensor phi     = data -> get_data_node("phi") -> clone(); 
    torch::Tensor energy  = data -> get_data_node("energy") -> clone(); 
    torch::Tensor pmc     = transform::cuda::PxPyPzE(pt, eta, phi, energy); 

    torch::Tensor met     = data -> get_data_graph("met") -> clone(); 
    torch::Tensor met_phi = data -> get_data_graph("phi") -> clone();     
    torch::Tensor met_xy  = torch::cat({
            transform::cuda::Px(met, met_phi), 
            transform::cuda::Py(met, met_phi)
    }, {-1});

    torch::Tensor truth = data -> get_truth_edge("top_edge") -> clone(); 
    torch::Tensor src = data -> edge_index -> index({0}).view({-1, 1}); 
    torch::Tensor dst = data -> edge_index -> index({1}).view({-1, 1}); 


    torch::Tensor matrix = torch::ones({pmc.size({0}), pmc.size({0})}, src.device()); 
    this -> _matrix = &matrix; 

    torch::Tensor trk_i = torch::ones_like(pt).to(torch::kLong).cumsum({0})-1; 
    this -> _path = &trk_i; 

    std::vector<torch::Tensor> H0 = {}; 
    for (size_t x(0); x < this -> _hidden; ++x){H0.push_back(torch::zeros_like(pt));}
    torch::Tensor H = torch::cat(H0, {-1}).to(torch::kFloat32);  
    H = this -> aggregation(src, dst, pmc, H);

    this -> _matrix = nullptr; 
    this -> _path   = nullptr; 
    this -> prediction_edge_feature("top_edge", H); 
}

recursivegraphneuralnetwork::~recursivegraphneuralnetwork(){}
model_template* recursivegraphneuralnetwork::clone(){
    return new recursivegraphneuralnetwork(); 
}

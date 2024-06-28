#include <RecursiveGraphNeuralNetwork.h>
#include <transform/cartesian-cuda.h>
#include <physics/cartesian-cuda.h>
#include <transform/polar-cuda.h>
#include <nusol/nusol-cuda.h>
#include <graph/graph-cuda.h>


recursivegraphneuralnetwork::recursivegraphneuralnetwork(){

    this -> rnn_dx = new torch::nn::Sequential({
            {"rnn_dx_l1", torch::nn::Linear(this -> _dx + 2*this -> _rep, this -> _hidden)},
            {"rnn_dx_relu1", torch::nn::ReLU()},
            {"rnn_dx_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_dx_l2", torch::nn::Linear(this -> _hidden, this -> _rep)}
    }); 

    this -> rnn_x = new torch::nn::Sequential({
            {"rnn_x_l1", torch::nn::Linear(this -> _x + this -> _rep, this -> _hidden)},
            {"rnn_x_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_x_l2", torch::nn::Linear(this -> _hidden, this -> _rep)}
    }); 

    this -> rnn_merge = new torch::nn::Sequential({
            {"rnn_merge_l1", torch::nn::Linear(this -> _rep*2, this -> _rep)},
            {"rnn_merge_relu1", torch::nn::ReLU()},
            {"rnn_merge_l2", torch::nn::Linear(this -> _rep, this -> _output)}
    }); 

    this -> rnn_update = new torch::nn::Sequential({
            {"rnn_update_l1", torch::nn::Linear(this -> _output*2, this -> _output)},
            {"rnn_update_relu1", torch::nn::ReLU()},
            {"rnn_update_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _output}))}, 
            {"rnn_update_l2", torch::nn::Linear(this -> _output, this -> _output)}
    }); 


    this -> register_module(this -> rnn_x); 
    this -> register_module(this -> rnn_dx); 
    this -> register_module(this -> rnn_merge); 
    this -> register_module(this -> rnn_update); 
}

torch::Tensor recursivegraphneuralnetwork::message(
        torch::Tensor _trk_i, torch::Tensor _trk_j, 
        torch::Tensor pmc, torch::Tensor pmc_i, torch::Tensor pmc_j,
        torch::Tensor hx_i, torch::Tensor hx_j

){
    torch::Tensor trk_ij = torch::cat({_trk_i, _trk_j}, {-1}); 

    torch::Tensor pmc_i_ = std::get<0>(graph::cuda::unique_aggregation(_trk_i, pmc)); 
    torch::Tensor pmc_j_ = std::get<0>(graph::cuda::unique_aggregation(_trk_j, pmc)); 
    torch::Tensor pmc_ij = std::get<0>(graph::cuda::unique_aggregation(trk_ij, pmc)); 
   
    torch::Tensor m_i  = physics::cuda::cartesian::M(pmc_i);
    torch::Tensor m_j  = physics::cuda::cartesian::M(pmc_j); 
    torch::Tensor m_ij = physics::cuda::cartesian::M(pmc_ij);
   
    torch::Tensor m_i_  = physics::cuda::cartesian::M(pmc_i_);
    torch::Tensor m_j_  = physics::cuda::cartesian::M(pmc_j_); 

    std::vector<torch::Tensor> dx_ = {m_j, m_i - m_j, m_j_, m_i_ - m_j_, hx_j, hx_i - hx_j}; 
    torch::Tensor hdx = (*this -> rnn_dx) -> forward(torch::cat(dx_, {-1}).to(torch::kFloat32)) + hx_j; 

    std::vector<torch::Tensor> x_ = {m_ij, hx_i}; 
    torch::Tensor hx = (*this -> rnn_x) -> forward(torch::cat(x_, {-1}).to(torch::kFloat32)) + hx_i;

    torch::Tensor id = torch::cat({hx, hdx - hx}, {-1}); 
    return (*this -> rnn_merge) -> forward(id); 
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

    torch::Tensor edge_index = data -> edge_index -> to(torch::kInt); 
    torch::Tensor src = edge_index.index({0}).view({-1}); 
    torch::Tensor dst = edge_index.index({1}).view({-1}); 


    std::vector<torch::Tensor> x_ = {}; 
    torch::Tensor nulls = torch::zeros_like(pt); 
    for (size_t t(0); t < this -> _rep; ++t){x_.push_back(nulls);}
    nulls = torch::cat(x_, {-1}).to(torch::kFloat32); 

    x_ = {physics::cuda::cartesian::M(pmc), nulls}; 
    torch::Tensor hx = (*this -> rnn_x) -> forward(torch::cat(x_, {-1}).to(torch::kFloat32));

    torch::Tensor trk = (torch::ones_like(pt).cumsum({0}) - 1).to(torch::kInt); 
    torch::Tensor idx_mlp = (torch::ones_like(src).cumsum({-1})-1).to(torch::kInt); 
    torch::Tensor idx_mat = torch::zeros({trk.size({0}), trk.size({0})}, src.device()).to(torch::kInt); 
    idx_mat.index_put_({src, dst}, idx_mlp); 

    int iter = 0; 
    torch::Tensor H_; 
    torch::Tensor edge_index_ = edge_index.clone(); 
    while(true){
        if (!edge_index_.size({1})){break;}
        torch::Tensor src_ = edge_index_.index({0}); 
        torch::Tensor dst_ = edge_index_.index({1}); 

        torch::Tensor idx = idx_mat.index({idx_mlp.index({src_}), idx_mlp.index({dst_})}); 

        torch::Tensor H = this -> message(
                trk.index({src_}), trk.index({dst_}), 
                pmc, pmc.index({src_}), pmc.index({dst_}), 
                hx.index({src_}), hx.index({dst_})
        ); 

        if (!iter){H_ = torch::zeros_like(H); ++iter;}

        torch::Tensor T_ = H_.index({idx});  
        torch::Tensor G  = (*this -> rnn_update) -> forward(torch::cat({H, T_ - H}, {-1})) + T_ - H; 
        torch::Tensor G_ = H_.clone(); 
        G_.index_put_({idx}, G); 
        H_ = (H_ + G_).softmax(-1)*G_ + (H_ + G_).softmax(-1)*H_;

        torch::Tensor sel = std::get<1>(G.max({-1})); 
        if (!sel.index({sel == 1}).size({0})){break;}
        edge_index_ = edge_index_.index({torch::indexing::Slice(), sel != 1}); 
        std::vector<torch::Tensor> gr_ = graph::cuda::edge_aggregation(edge_index.to(torch::kLong), H_, pmc)["1"]; 

        trk  = gr_[0].index({gr_[2]});
        torch::Tensor pmc_ = gr_[3]; 
        x_ = {physics::cuda::cartesian::M(pmc_), hx}; 
        torch::Tensor hx_ = (*this -> rnn_x) -> forward(torch::cat(x_, {-1}).to(torch::kFloat32)) + hx;
        hx = hx_; 

    }
    this -> prediction_edge_feature("top_edge", H_); 
}

recursivegraphneuralnetwork::~recursivegraphneuralnetwork(){}
model_template* recursivegraphneuralnetwork::clone(){
    return new recursivegraphneuralnetwork(); 
}

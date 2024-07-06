#include <RecursiveGraphNeuralNetwork.h>
#include <transform/cartesian-cuda.h>
#include <physics/cartesian-cuda.h>
#include <transform/polar-cuda.h>
#include <nusol/nusol-cuda.h>
#include <graph/graph-cuda.h>


recursivegraphneuralnetwork::recursivegraphneuralnetwork(){

    this -> rnn_dx = new torch::nn::Sequential({
            {"rnn_dx_l1", torch::nn::Linear(this -> _dx + 2*this -> _rep, this -> _rep*2)},
            {"rnn_dx_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep*2}))}, 
            {"rnn_dx_s1", torch::nn::Sigmoid()},
            {"rnn_dx_r1", torch::nn::ReLU()},
            {"rnn_dx_l2", torch::nn::Linear(this -> _rep*2, this -> _rep)},
            {"rnn_dx_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep}))}, 
            {"rnn_dx_s2", torch::nn::Sigmoid()},
            {"rnn_dx_r2", torch::nn::ReLU()},
            {"rnn_dx_l3", torch::nn::Linear(this -> _rep, this -> _rep)}
    }); 

    this -> rnn_x = new torch::nn::Sequential({
            {"rnn_x_l1", torch::nn::Linear(this -> _x + this -> _rep, this -> _rep)},
            {"rnn_x_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep}))}, 
            {"rnn_x_l2", torch::nn::Linear(this -> _rep, this -> _rep)},
            {"rnn_x_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep}))}, 
            {"rnn_x_l3", torch::nn::Linear(this -> _rep, this -> _rep)}
    }); 

    this -> rnn_merge = new torch::nn::Sequential({
            {"rnn_mrg_l1", torch::nn::Linear(this -> _rep*3, this -> _rep*3)},
            {"rnn_mrg_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep*3}))}, 
            {"rnn_mrg_s1", torch::nn::Sigmoid()},
            {"rnn_mrg_r1", torch::nn::ReLU()},
            {"rnn_mrg_l2", torch::nn::Linear(this -> _rep*3, this -> _rep*2)},
            {"rnn_mrg_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep*2}))}, 
            {"rnn_mrg_s2", torch::nn::Sigmoid()},
            {"rnn_mrg_r2", torch::nn::ReLU()},
            {"rnn_mrg_l3", torch::nn::Linear(this -> _rep*2, this -> _rep)}
    }); 

    this -> rnn_update = new torch::nn::Sequential({
            {"rnn_up_l1", torch::nn::Linear(this -> _output*2 + this -> _rep*2, this -> _rep*2)},
            {"rnn_up_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep*2}))}, 
            {"rnn_up_s1", torch::nn::Sigmoid()},
            {"rnn_up_r1", torch::nn::ReLU()},
            {"rnn_up_l2", torch::nn::Linear(this -> _rep*2, this -> _rep)},
            {"rnn_up_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep}))}, 
            {"rnn_up_s2", torch::nn::Sigmoid()},
            {"rnn_up_r2", torch::nn::ReLU()},
            {"rnn_up_l3", torch::nn::Linear(this -> _rep, this -> _output)}
    }); 


    this -> register_module(this -> rnn_x); 
    this -> register_module(this -> rnn_dx); 
    this -> register_module(this -> rnn_merge); 
    this -> register_module(this -> rnn_update); 
}

torch::Tensor recursivegraphneuralnetwork::neutrino(
        torch::Tensor edge_index, torch::Tensor pmc, 
        torch::Tensor pid, torch::Tensor met_xy, 
        torch::Tensor batch, std::string* hash
){
    if (!this -> NuR){return pmc;}
    if (this -> _cache.count(*hash)){return this -> _cache[*hash];}

    std::map<std::string, torch::Tensor> nus = nusol::cuda::combinatorial(
        edge_index, batch, pmc, pid, met_xy, 172.62*1000, 80.385*1000, 0.0, 0.9, 0.9, 1e-10
    ); 
    torch::Tensor combi = nus["combi"].sum({-1}) > 0;
    if (combi.index({combi}).size({0})){return pmc;}

    torch::Tensor nu1 = nus["combi"].index({combi, 2}); 
    torch::Tensor nu2 = nus["combi"].index({combi, 3}); 
    pmc.index_put_({nu1}, nus["nu_1f"] + pmc.index({nu1})); 
    pmc.index_put_({nu2}, nus["nu_2f"] + pmc.index({nu2}));
    this -> _cache[*hash] = pmc; 
    return pmc; 
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

    torch::Tensor dr   = physics::cuda::cartesian::DeltaR(pmc_i, pmc_j); 

    std::vector<torch::Tensor> dx_ = {
        m_ij, pmc_ij, dr, 
        m_i, m_j - m_i, m_i_, m_j_ - m_i_, 
        pmc_i, pmc_j - pmc_i, pmc_i_, pmc_j_ - pmc_i_, 
        hx_i, hx_j - hx_i}; 
    torch::Tensor hdx = (*this -> rnn_dx) -> forward(torch::cat(dx_, {-1}).to(torch::kFloat32)); 

    torch::Tensor id = torch::cat({hdx, hx_i, hx_j - hx_i}, {-1}); 
    return (*this -> rnn_merge) -> forward(id); 
}

void recursivegraphneuralnetwork::forward(graph_t* data){

    torch::Tensor pt         = data -> get_data_node("pt") -> clone();
    torch::Tensor eta        = data -> get_data_node("eta") -> clone(); 
    torch::Tensor phi        = data -> get_data_node("phi") -> clone(); 
    torch::Tensor energy     = data -> get_data_node("energy") -> clone(); 
    torch::Tensor met        = data -> get_data_graph("met") -> clone(); 
    torch::Tensor met_phi    = data -> get_data_graph("phi") -> clone();
    torch::Tensor is_lep     = data -> get_data_node("is_lep") -> clone(); 
    torch::Tensor is_b       = data -> get_data_node("is_b") -> clone(); 
    torch::Tensor edge_index = data -> edge_index -> to(torch::kLong); 

    torch::Tensor batch      = torch::zeros_like(pt.view({-1})).to(torch::kLong); 
    torch::Tensor pmc        = transform::cuda::PxPyPzE(pt, eta, phi, energy); 
    torch::Tensor pid        = torch::cat({is_lep, is_b}, {-1}); 
    torch::Tensor met_xy     = torch::cat({transform::cuda::Px(met, met_phi), transform::cuda::Py(met, met_phi)}, {-1});
    pmc = this -> neutrino(edge_index, pmc, pid, met_xy, batch, data -> hash);

    torch::Tensor src = edge_index.index({0}).view({-1}); 
    torch::Tensor dst = edge_index.index({1}).view({-1}); 

    std::vector<torch::Tensor> x_ = {}; 
    torch::Tensor nulls = torch::zeros_like(pt); 
    for (size_t t(0); t < this -> _rep; ++t){x_.push_back(nulls);}
    nulls = torch::cat(x_, {-1}).to(torch::kFloat32); 

    x_ = {physics::cuda::cartesian::M(pmc), pmc, nulls}; 
    torch::Tensor hx = (*this -> rnn_x) -> forward(torch::cat(x_, {-1}).to(torch::kFloat32));

    torch::Tensor trk = (torch::ones_like(pt).cumsum({0}) - 1).to(torch::kInt); 
    torch::Tensor idx_mlp = (torch::ones_like(src).cumsum({-1})-1).to(torch::kInt); 
    torch::Tensor idx_mat = torch::zeros({trk.size({0}), trk.size({0})}, src.device()).to(torch::kInt); 
    idx_mat.index_put_({src, dst}, idx_mlp); 

    int iter = 0; 
    torch::Tensor H_;
    torch::Tensor edge_index_ = edge_index.clone(); 
    torch::Tensor G_ = torch::zeros_like(torch::cat({src.view({-1, 1}), dst.view({-1, 1})}, {-1})).to(torch::kFloat32); 
    torch::Tensor G  = torch::zeros_like(torch::cat({src.view({-1, 1}), dst.view({-1, 1})}, {-1})).to(torch::kFloat32); 
    while(true){
        if (!edge_index_.size({1})){break;}
        torch::Tensor src_ = edge_index_.index({0}); 
        torch::Tensor dst_ = edge_index_.index({1}); 
        torch::Tensor H = this -> message(
                trk.index({src_}), trk.index({dst_}), 
                pmc, pmc.index({src_}), pmc.index({dst_}), 
                hx.index({src_}), hx.index({dst_})
        ); 

        if (!iter){H_ = torch::zeros_like(H); ++iter;}
        torch::Tensor idx = idx_mat.index({idx_mlp.index({src_}), idx_mlp.index({dst_})}); 
        G  = (*this -> rnn_update) -> forward(torch::cat({H, H_ - H, G, G_.index({idx}) - G}, {-1})); 
        G_.index_put_({idx}, G * G.softmax(-1)); 

        torch::Tensor sel = std::get<1>(G.max({-1})); 
        if (!sel.index({sel == 1}).size({0})){break;}
        G  = G.index({sel != 1}); 
        H_ = H.index({sel != 1}); 

        edge_index_ = edge_index_.index({torch::indexing::Slice(), sel != 1}); 
        std::vector<torch::Tensor> gr_ = graph::cuda::edge_aggregation(edge_index, G_, pmc)["1"]; 

        trk  = gr_[0].index({gr_[2]});
        torch::Tensor pmc_ = gr_[3]; 
        x_ = {physics::cuda::cartesian::M(pmc_), pmc_, hx}; 
        hx = (*this -> rnn_x) -> forward(torch::cat(x_, {-1}).to(torch::kFloat32));
    }
    this -> prediction_edge_feature("top_edge", G_); 
}

recursivegraphneuralnetwork::~recursivegraphneuralnetwork(){}
model_template* recursivegraphneuralnetwork::clone(){
    return new recursivegraphneuralnetwork(); 
}

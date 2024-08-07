#include <Experimental.h>
#include <transform/cartesian-cuda.h>
#include <physics/cartesian-cuda.h>
#include <transform/polar-cuda.h>
#include <graph/graph-cuda.h>


experimental::experimental(int rep, double drop_out){
    this -> drop_out = drop_out; 
    this -> _rep = rep; 

    this -> rnn_dx = new torch::nn::Sequential({
            {"rnn_dx_l1", torch::nn::Linear(2*this -> _rep, this -> _rep*2)},
            {"rnn_dx_r1" , torch::nn::SiLU()},
            {"rnn_dx_l3", torch::nn::Linear(this -> _rep*2, this -> _rep)}, 
    }); 

    this -> rnn_x = new torch::nn::Sequential({
            {"rnn_x_l1", torch::nn::Linear(this -> _x + this -> _rep, this -> _hidden)},
            {"rnn_x_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_x_l2", torch::nn::Linear(this -> _hidden, this -> _rep)}
    }); 

    this -> rnn_merge = new torch::nn::Sequential({
            {"rnn_mrg_l1", torch::nn::Linear(this -> _rep*3, this -> _rep)},
            {"rnn_mrg_r1" , torch::nn::ReLU()},
            {"rnn_mrg_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep}))}, 
            {"rnn_mrg_l2", torch::nn::Linear(this -> _rep, this -> _rep)}
    }); 

    this -> rnn_update = new torch::nn::Sequential({
            {"rnn_up_l1" , torch::nn::Linear(this -> _output*2 + this -> _rep*2, this -> _rep*2)},
            {"rnn_up_n1" , torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep*2}))}, 
            {"rnn_up_l3" , torch::nn::Linear(this -> _rep*2, this -> _output)}
    }); 

    this -> rnn_exotic = new torch::nn::Sequential({
            {"exotic_l1" , torch::nn::Linear(this -> _x + this -> _output, this -> _rep)}, 
            {"exotic_n1" , torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _rep}))},
            {"exotic_l2" , torch::nn::Linear(this -> _rep, this -> _output)}
    });

    this -> node_aggr_mlp = new torch::nn::Sequential({
            {"node_aggr_l1" , torch::nn::Linear(this -> _x, this -> _x)}, 
            {"node_aggr_n1" , torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _x}))},
            {"node_aggr_l2" , torch::nn::Linear(this -> _x, this -> _x)}
    }); 

    this -> ntops_mlp = new torch::nn::Sequential({
            {"ntops_l1" , torch::nn::Linear(this -> _x + 4, this -> _x+4)}, 
            {"ntops_n1" , torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _x+4}))},
            {"ntops_sl1", torch::nn::SiLU()},
            {"ntops_l2" , torch::nn::Linear(this -> _x+4, this -> _x)}, 
            {"ntops_n2" , torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _x}))},
            {"ntops_l3" , torch::nn::Linear(this -> _x, this -> _x)}
    }); 

    int exo_mlp_ = this -> _x + this -> _output; 
    this -> exo_mlp = new torch::nn::Sequential({
            {"exo_l1", torch::nn::Linear(exo_mlp_, exo_mlp_)}, 
            {"exo_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({exo_mlp_}))},
            {"exo_sl1", torch::nn::SiLU()},
            {"exo_l2" , torch::nn::Linear(exo_mlp_, exo_mlp_)}, 
            {"exo_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({exo_mlp_}))},
            {"exo_l3", torch::nn::Linear(exo_mlp_, this -> _output)}
    }); 

    this -> register_module(this -> rnn_x,         mlp_init::uniform); 
    this -> register_module(this -> rnn_dx,        mlp_init::uniform); 
    this -> register_module(this -> rnn_merge,     mlp_init::uniform); 
    this -> register_module(this -> rnn_update,    mlp_init::uniform); 
    this -> register_module(this -> rnn_exotic,    mlp_init::uniform); 

    this -> register_module(this -> node_aggr_mlp, mlp_init::uniform); 
    this -> register_module(this -> ntops_mlp,     mlp_init::uniform); 
    this -> register_module(this -> exo_mlp,       mlp_init::uniform); 
}

torch::Tensor experimental::message(
        torch::Tensor _trk_i, torch::Tensor _trk_j, torch::Tensor pmc, torch::Tensor hx_i, torch::Tensor hx_j
){
    torch::Tensor trk_ij = torch::cat({_trk_i, _trk_j}, {-1}); 
    torch::Tensor pmc_ij = std::get<0>(graph::cuda::unique_aggregation(trk_ij, pmc)); 
    torch::Tensor m_ij = physics::cuda::cartesian::M(pmc_ij);

    torch::Tensor hx = torch::cat({m_ij, pmc_ij, hx_i}, {-1}); 
    hx = (*this -> rnn_x) -> forward(hx.to(torch::kFloat32)); 
    torch::Tensor hdx = (*this -> rnn_dx) -> forward(torch::cat({hx, hx - hx_i}, {-1}).to(torch::kFloat32)); 
    return (*this -> rnn_merge) -> forward(torch::cat({hdx, hx_i, hx_i - hx_j}, {-1})) + hdx; 
}

void experimental::forward(graph_t* data){

    // get the particle 4-vector and convert it to cartesian
    torch::Tensor pt     = data -> get_data_node("pt", this) -> clone();
    torch::Tensor eta    = data -> get_data_node("eta", this) -> clone();
    torch::Tensor phi    = data -> get_data_node("phi", this) -> clone();
    torch::Tensor energy = data -> get_data_node("energy", this) -> clone();
    torch::Tensor pmc    = transform::cuda::PxPyPzE(pt, eta, phi, energy); 

    // the event topology
    torch::Tensor edge_index = data -> get_edge_index(this) -> to(torch::kLong); 
    torch::Tensor num_jets   = data -> get_data_graph("num_jets", this) -> clone(); 
    torch::Tensor num_leps   = data -> get_data_graph("num_leps", this) -> clone(); 
    torch::Tensor met        = data -> get_data_graph("met", this) -> clone(); 
    torch::Tensor met_phi    = data -> get_data_graph("phi", this) -> clone();

    torch::Tensor metphi  = torch::cat({met, met_phi}, {-1});
    torch::Tensor src     = edge_index.index({0}).view({-1}); 
    torch::Tensor dst     = edge_index.index({1}).view({-1}); 

    // ------ create an empty buffer for the recursion ------- //
    std::vector<torch::Tensor> x_ = {}; 
    torch::Tensor nulls = torch::zeros_like(pt); 
    for (size_t t(0); t < this -> _rep; ++t){x_.push_back(nulls);}
    nulls = torch::cat(x_, {-1}).to(torch::kFloat32); 

    // ------ Create an initial prediction with the edge classifier ------ //
    nulls = torch::cat({physics::cuda::cartesian::M(pmc), pmc, nulls}, {-1}).to(torch::kFloat32); 
    torch::Tensor hx = (*this -> rnn_x) -> forward(nulls);

    // ------ index the nodes from 0 to N-1 ----- //
    torch::Tensor trk = (torch::ones_like(pt).cumsum({0}) - 1).to(torch::kInt); 

    // ------ index the edges from 0 to N^2 -1 ------ //
    torch::Tensor idx_mlp = (torch::ones_like(src).cumsum({-1})-1).to(torch::kInt); 
    torch::Tensor idx_mat = torch::zeros({trk.size({0}), trk.size({0})}, src.device()).to(torch::kInt); 
    idx_mat.index_put_({src, dst}, idx_mlp); 

    // ------ declare the intermediate states ------- //
    int iter = 0; 
    torch::Tensor H_, T;
    torch::Tensor edge_index_ = edge_index.clone(); 
    torch::Tensor G_ = torch::zeros_like(torch::cat({src.view({-1, 1}), dst.view({-1, 1})}, {-1})).to(torch::kFloat32); 
    torch::Tensor G  = torch::zeros_like(torch::cat({src.view({-1, 1}), dst.view({-1, 1})}, {-1})).to(torch::kFloat32); 
    while(true){

        // ---- check if the current edge index prediction is null ---- //
        if (!edge_index_.size({1})){break;}

        // ----- get the source and destination indices ---- //
        torch::Tensor src_ = edge_index_.index({0}); 
        torch::Tensor dst_ = edge_index_.index({1}); 

        // ----- Make a new prediction with prior inputs of hx (prior edge prediction) ---- //
        torch::Tensor H = this -> message(trk.index({src_}), trk.index({dst_}), pmc, hx.index({src_}), hx.index({dst_})); 

        // ----- if this is the first iteration, initialize H_ ----- //
        if (!iter){H_ = torch::zeros_like(H); ++iter;}

        // ----- use the index matrix to map the source and destination edges to the edge index ----- //
        torch::Tensor idx = idx_mat.index({idx_mlp.index({src_}), idx_mlp.index({dst_})}); 

        // ----- make new prediction of G based on H, H_, prior G, and the current G_ state ---- //
        G  = (*this -> rnn_update) -> forward(torch::cat({H, H_ - H, G, G_.index({idx}) - G}, {-1})); 

        // ----- scale the output by softmax ----- //
        G_.index_put_({idx}, G); 
        T = G_; 
        // ----- use new G prediction to get new topology prediction ---- //
        torch::Tensor sel = std::get<1>(G.max({-1})); 

        // ---- check if the new prediction is simply null ---- /
        if (!sel.index({sel == 1}).size({0})){break;}
        G  = G.index({sel != 1}); 
        H_ = H.index({sel != 1}); 

        // ----- update the topology state ---- //
        edge_index_ = edge_index_.index({torch::indexing::Slice(), sel != 1}); 

        // ----- create a new intermediate state of the node ----- //
        std::vector<torch::Tensor> gr_ = graph::cuda::edge_aggregation(edge_index, G_, pmc)["1"]; 

        trk  = gr_[0].index({gr_[2]});
        torch::Tensor pmc_ = gr_[3]; 
        nulls = torch::cat({physics::cuda::cartesian::M(pmc_), pmc_, hx}, {-1}); 
        hx = (*this -> rnn_x) -> forward(nulls.to(torch::kFloat32));
    }

    // ----------- count the number of tops and use them for Z/H boson ---------- //
    std::vector<torch::Tensor> gr_clust = graph::cuda::edge_aggregation(edge_index, T, pmc)["1"];  

    // ----------- reverse the clustering on a per node basis ----------- //
    torch::Tensor clusters = gr_clust[0].index({gr_clust[2]}); 
    torch::Tensor c_ij  = torch::cat({clusters.index({src}), clusters.index({dst})}, {-1}); 
   
    // ----------- perform a single edge update based on clusters ----------- //
    torch::Tensor z_pmc = std::get<0>(graph::cuda::unique_aggregation(c_ij, pmc));
    torch::Tensor z_inv = physics::cuda::cartesian::M(z_pmc) - this -> res_mass;  
    torch::Tensor res_edge = (*this -> rnn_exotic) -> forward(torch::cat({z_inv, z_pmc, T}, {-1}).to(torch::kFloat32))*T.softmax(-1);

    // ---------- compress edge details to node features ------------ //
    torch::Tensor top_matrix_0 = torch::zeros_like(idx_mat).to(torch::kFloat32); 
    top_matrix_0.index_put_({src, dst}, T.index({torch::indexing::Slice(), 0})); 

    torch::Tensor top_matrix_1 = torch::zeros_like(idx_mat).to(torch::kFloat32); 
    top_matrix_1.index_put_({src, dst}, T.index({torch::indexing::Slice(), 1})); 

    torch::Tensor top_matrix = torch::cat({top_matrix_0.sum({-1}, true), top_matrix_1.sum({-1}, true)}, {-1}); 
    top_matrix = torch::cat({top_matrix, physics::cuda::cartesian::M(gr_clust[3]) - torch::ones_like(pt)*172.62*1000}, {-1}); 
    top_matrix = torch::cat({top_matrix, (clusters > -1).sum({-1}, true), torch::ones_like(pt)*gr_clust[0].size({0})}, {-1}); 
    top_matrix = (*this -> node_aggr_mlp) -> forward(top_matrix.to(torch::kFloat32)); 

    // ---------- compress node details to graph -------- //
    top_matrix = top_matrix.sum({0}, true); 
    top_matrix = torch::cat({top_matrix, num_jets, num_leps, metphi}, {-1}); 
    torch::Tensor ntops  = (*this -> ntops_mlp) -> forward(top_matrix.to(torch::kFloat32));  
    torch::Tensor is_res = (*this -> exo_mlp) -> forward(torch::cat({ntops, (res_edge.softmax(-1)*res_edge).sum({0}, true)}, {-1})); 

    this -> prediction_graph_feature("signal" , is_res); 
    this -> prediction_graph_feature("ntops"  , ntops); 
    this -> prediction_edge_feature("top_edge", T); 
    this -> prediction_edge_feature("res_edge", res_edge); 

    if (!this -> inference_mode){return;}
    this -> prediction_extra("top_edge_score", T.softmax(-1));
    this -> prediction_extra("ntops_score"   , ntops.softmax(-1)); 

    this -> prediction_extra("res_edge_score", res_edge.softmax(-1));
    this -> prediction_extra("is_res_score"  , is_res.softmax(-1)); 

    torch::Tensor top_pred = graph::cuda::edge_aggregation(edge_index, T, pmc)["1"][1]; 
    this -> prediction_extra("top_pmc", top_pred); 

    if (!this -> is_mc){return;}
    torch::Tensor truth_t = data -> get_truth_edge("top_edge", this) -> view({-1}); 
    torch::Tensor truth_ = torch::zeros_like(T); 
    for (int x(0); x < T.size({-1}); ++x){truth_.index_put_({truth_t == x, x}, 1);}
    torch::Tensor truth_top = graph::cuda::edge_aggregation(edge_index, truth_, pmc)["1"][1]; 
    this -> prediction_extra("truth_top_pmc", truth_top); 
}

experimental::~experimental(){}
model_template* experimental::clone(){
    experimental* rnn = new experimental(this -> _rep, this -> drop_out); 
    rnn -> _dx      = this -> _dx;      
    rnn -> _x       = this -> _x;       
    rnn -> _output  = this -> _output;  
    rnn -> res_mass = this -> res_mass; 
    rnn -> is_mc    = this -> is_mc;
    return rnn;  
}

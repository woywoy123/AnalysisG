#include <Experimental.h>
#include <transform/cartesian-cuda.h>
#include <physics/cartesian-cuda.h>
#include <transform/polar-cuda.h>
#include <graph/graph-cuda.h>


experimental::experimental(){
    this -> rnn_x = new torch::nn::Sequential({
            {"rnn_x_l1", torch::nn::Linear(this -> _xin*2, this -> _hidden)},
            {"rnn_x_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_x_t1", torch::nn::Tanh()},
            {"rnn_x_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)},
            {"rnn_x_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_x_r2", torch::nn::ReLU()},
            {"rnn_x_l3", torch::nn::Linear(this -> _hidden, this -> _xin)}
    }); 

    int dxx = this -> _dxin*2 + this -> _xin*2; 
    this -> rnn_dx = new torch::nn::Sequential({
            {"rnn_dx_l1", torch::nn::Linear(dxx, this -> _hidden)}, 
            {"rnn_dx_t1", torch::nn::Tanh()},
            {"rnn_dx_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_dx_r1", torch::nn::ReLU()},
            {"rnn_dx_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)}, 
            {"rnn_dx_t2", torch::nn::Tanh()},
            {"rnn_dx_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_dx_r2", torch::nn::ReLU()},
            {"rnn_dx_l3", torch::nn::Linear(this -> _hidden, this -> _xin)} 
    }); 

    this -> rnn_merge = new torch::nn::Sequential({
            {"rnn_mrg_l1", torch::nn::Linear(this -> _xin*2, this -> _hidden)}, 
            {"rnn_mrg_t1", torch::nn::Tanh()},
            {"rnn_mrg_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_mrg_r1", torch::nn::ReLU()},
            {"rnn_mrg_l2", torch::nn::Linear(this -> _hidden, this -> _xin)}
    }); 

    dxx = this -> _xin*2 + this -> _xout*2; 
    this -> rnn_top_edge = new torch::nn::Sequential({
            {"rnn_top_l1", torch::nn::Linear(dxx, this -> _xout*2)}, 
            {"rnn_top_t1", torch::nn::Tanh()},
            {"rnn_top_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xout*2}))}, 
            {"rnn_top_r1", torch::nn::ReLU()},
            {"rnn_top_l2", torch::nn::Linear(this -> _xout*2, this -> _xout)}
    }); 

    dxx += this -> _xout*2; 
    this -> rnn_res_edge = new torch::nn::Sequential({
            {"rnn_res_l1", torch::nn::Linear(dxx, this -> _xout*2)}, 
            {"rnn_res_t1", torch::nn::Tanh()},
            {"rnn_res_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xout*2}))}, 
            {"rnn_res_r1", torch::nn::ReLU()},
            {"rnn_res_l2", torch::nn::Linear(this -> _xout*2, this -> _xout)}
    }); 

    this -> mlp_ntop = new torch::nn::Sequential({
            {"rnn_ntop_l1", torch::nn::Linear(this -> _xin, this -> _hidden)}, 
            {"rnn_ntop_t1", torch::nn::Tanh()},
            {"rnn_ntop_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_ntop_r1", torch::nn::ReLU()},
            {"rnn_ntop_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)}
    }); 

    this -> mlp_sig = new torch::nn::Sequential({
            {"rnn_sig_l1", torch::nn::Linear(this -> _xin, this -> _hidden)}, 
            {"rnn_sig_t1", torch::nn::Tanh()},
            {"rnn_sig_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_sig_r1", torch::nn::ReLU()},
            {"rnn_sig_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)}
    }); 

    this -> register_module(this -> rnn_x); 
    this -> register_module(this -> rnn_dx); 
    this -> register_module(this -> rnn_merge); 
    this -> register_module(this -> rnn_top_edge); 
    this -> register_module(this -> rnn_res_edge); 
    this -> register_module(this -> mlp_ntop); 
    this -> register_module(this -> mlp_sig); 

}

torch::Tensor experimental::message(
        torch::Tensor _trk_i, torch::Tensor _trk_j, torch::Tensor pmc, torch::Tensor hx_i, torch::Tensor hx_j
){
    torch::Tensor trk_ij = torch::cat({_trk_i, _trk_j}, {-1}); 

    std::tuple<torch::Tensor, torch::Tensor> aggr = graph::cuda::unique_aggregation(trk_ij, pmc); 
    torch::Tensor pmc_ij = std::get<0>(aggr); 
    torch::Tensor nds_ij = (std::get<1>(aggr) > -1).sum({-1}).view({-1, 1}); 
    torch::Tensor m_ij   = physics::cuda::cartesian::M(pmc_ij);
    torch::Tensor fx_ij  = torch::cat({m_ij, pmc_ij, nds_ij}, {-1}).to(torch::kFloat32); 
    fx_ij = (*this -> rnn_x) -> forward(torch::cat({fx_ij, hx_i}, {-1})) + fx_ij; 

    torch::Tensor fx_i  = std::get<0>(graph::cuda::unique_aggregation(_trk_i, pmc)); 
    torch::Tensor fx_j  = std::get<0>(graph::cuda::unique_aggregation(_trk_j, pmc)); 

    fx_i = torch::cat({fx_i, physics::cuda::cartesian::M(fx_i)}, {-1}).to(torch::kFloat32); 
    fx_j = torch::cat({fx_j, physics::cuda::cartesian::M(fx_j)}, {-1}).to(torch::kFloat32); 

    torch::Tensor dx = (*this -> rnn_dx) -> forward(torch::cat({fx_i, fx_j - fx_i, hx_i, hx_j - hx_i}, {-1})); 
    return (fx_ij - (*this -> rnn_merge) -> forward(torch::cat({fx_ij, dx}, {-1})))*dx; 
}

void experimental::forward(graph_t* data){

    // get the particle 4-vector and convert it to cartesian
    torch::Tensor pt     = data -> get_data_node("pt", this) -> clone();
    torch::Tensor eta    = data -> get_data_node("eta", this) -> clone();
    torch::Tensor phi    = data -> get_data_node("phi", this) -> clone();
    torch::Tensor energy = data -> get_data_node("energy", this) -> clone();
    torch::Tensor pmc    = transform::cuda::PxPyPzE(pt, eta, phi, energy) / 1000.0; 

    torch::Tensor edge_index = data -> get_edge_index(this) -> to(torch::kLong); 
    torch::Tensor src        = edge_index.index({0}).view({-1}); 
    torch::Tensor dst        = edge_index.index({1}).view({-1}); 

    // the event features
    torch::Tensor num_jets = data -> get_data_graph("num_jets", this) -> clone(); 
    torch::Tensor num_bjet = data -> get_data_node("is_b", this) -> sum({0}).view({-1, 1});  
    torch::Tensor num_leps = data -> get_data_graph("num_leps", this) -> clone(); 
    torch::Tensor met_phi  = data -> get_data_graph("phi", this) -> clone();
    torch::Tensor met      = data -> get_data_graph("met", this) -> clone() / 1000.0; 
    torch::Tensor pid = torch::cat({num_jets, num_bjet, num_leps, met, met_phi}, {-1});  

    // ------ index the nodes from 0 to N-1 ----- //
    torch::Tensor trk = (torch::ones_like(pt).cumsum({0}) - 1).to(torch::kInt); 
    torch::Tensor nds = (trk > -1).sum({-1}).view({-1, 1}); 

    // ------ index the edges from 0 to N^2 -1 ------ //
    torch::Tensor idx_mlp = torch::ones_like(src).cumsum({-1})-1; 
    torch::Tensor idx_mat = torch::zeros({trk.size({0}), trk.size({0})}, src.device()).to(torch::kLong); 
    idx_mat.index_put_({src, dst}, idx_mlp.to(torch::kLong)); 

     // ------ Create an initial prediction with the classifier ------ //
    torch::Tensor fx = torch::cat({physics::cuda::cartesian::M(pmc), pmc, nds}, {-1}).to(torch::kFloat32); 
    torch::Tensor hx = (*this -> rnn_x) -> forward(torch::cat({fx, torch::zeros_like(fx)}, {-1})) + fx;

    torch::Tensor hx_ij = this -> message(trk.index({src}), trk.index({dst}), pmc, hx.index({src}), hx.index({dst}));
    torch::Tensor hx_px = torch::cat({hx.index({src}), hx_ij}, {-1}); 

    torch::Tensor nx = torch::zeros_like(idx_mlp.view({-1, 1}));
    nx = torch::cat({nx, nx}, {-1}).to(torch::kFloat32); 

    torch::Tensor top_edge = (*this -> rnn_top_edge) -> forward(torch::cat({hx_px, nx, nx}, {-1})); 
    torch::Tensor res_edge = (*this -> rnn_res_edge) -> forward(torch::cat({hx_px, nx, nx, nx, nx}, {-1})); 

    torch::Tensor top_edge_ = top_edge.clone(); 
    torch::Tensor res_edge_ = res_edge.clone(); 

    std::vector<torch::Tensor> gr_; 
    torch::Tensor edge_index_ = edge_index.clone();  
    while (edge_index_.size({1})){

        torch::Tensor src_ = edge_index_.index({0}); 
        torch::Tensor dst_ = edge_index_.index({1});

        // ----- use the index matrix to map the source and destination edges to the edge index ----- //
        torch::Tensor idx = idx_mat.index({src_, dst_}); 

        // ----------- create a new intermediate state of the nodes ----------- //
        gr_ = graph::cuda::edge_aggregation(edge_index, top_edge, pmc)["1"]; 
        trk = gr_[0].index({gr_[2]});
        nds = (trk > -1).sum({-1}).view({-1, 1}); 
        fx  = torch::cat({physics::cuda::cartesian::M(gr_[3]), gr_[3], nds}, {-1}).to(torch::kFloat32); 
        hx  = (*this -> rnn_x) -> forward(torch::cat({fx, hx}, {-1})) + fx;

        // ------------------ check edges for new paths ---------------- //
        torch::Tensor hx_ij_ = this -> message(trk.index({src_}), trk.index({dst_}), pmc, hx.index({src_}), hx.index({dst_}));
        hx_ij.index_put_({idx}, hx_ij_); 

        torch::Tensor hx_px_ = torch::cat({hx.index({src_}), hx_ij_}, {-1}); 
        hx_px.index_put_({idx}, hx_px_); 

        hx_px_ = torch::cat({hx_px_, top_edge_, top_edge.index({idx}) - top_edge_}, {-1}); 
        top_edge_ = (*this -> rnn_top_edge) -> forward(hx_px_);

        hx_px_ = torch::cat({hx_px_, res_edge_, res_edge.index({idx}) - res_edge_}, {-1}); 
        res_edge_ = (*this -> rnn_res_edge) -> forward(hx_px_); 

        // ----- update the top_edge prediction weights by index ------- //
        top_edge.index_put_({idx}, top_edge_); 
        res_edge.index_put_({idx}, res_edge_); 
        torch::Tensor sel = std::get<1>(top_edge_.max({-1})); 

        // ---- check if the new prediction is simply null ---- /
        if (!sel.index({sel == 1}).size({0})){break;}
        edge_index_ = edge_index_.index({torch::indexing::Slice(), sel != 1}); 
        top_edge_   = top_edge_.index({sel != 1}); 
        res_edge_   = res_edge_.index({sel != 1}); 
    }
    
    // ----------- compress the top data ----------- //
    gr_ = graph::cuda::edge_aggregation(edge_index, top_edge, pmc)["1"];  
    trk = torch::cat({(gr_[0] > -1).sum({-1}, true), physics::cuda::cartesian::M(gr_[1]) - 172.68, gr_[1]}, {-1}); 
    torch::Tensor t_cmpr = (*this -> mlp_ntop) -> forward(trk.to(torch::kFloat32)); 
    torch::Tensor c_cmpr = torch::cat({hx_ij, hx_px, top_edge}, {-1}); 

    torch::Tensor top_max = std::get<0>(t_cmpr.softmax(0).max({0}))*std::get<0>(t_cmpr.max({0})); 
    torch::Tensor hx_max  = std::get<0>(c_cmpr.softmax({0}).max({0}))*std::get<0>(c_cmpr.max({0})); 

    torch::Tensor top_min = std::get<0>(t_cmpr.softmax(0).min({0}))*std::get<0>(t_cmpr.min({0}));
    torch::Tensor hx_min  = std::get<0>(c_cmpr.softmax({0}).min({0}))*std::get<0>(c_cmpr.min({0})); 

    torch::Tensor top_mean = t_cmpr.softmax(0).mean({0})*t_cmpr.mean({0}); 
    torch::Tensor hx_mean  = c_cmpr.softmax({0}).mean({0})*c_cmpr.mean({0}); 

    torch::Tensor _ntops_hx = torch::cat({hx_max.view({-1, 1}) , hx_min.view({-1, 1}) , hx_mean.view({-1, 1})} , {-1}); 
    torch::Tensor _ntops_tp = torch::cat({top_max.view({-1, 1}), top_min.view({-1, 1}), top_mean.view({-1, 1})}, {-1}); 
    torch::Tensor ntops = _ntops_tp.matmul(_ntops_hx.transpose(1, 0)).view({-1, 5}).sum({0}, true).softmax(-1); 

    gr_ = graph::cuda::edge_aggregation(edge_index, res_edge, pmc)["1"];  
    trk = torch::cat({(gr_[0] > -1).sum({-1}, true), physics::cuda::cartesian::M(gr_[1]), gr_[1]}, {-1}); 
    torch::Tensor rs_cmpr = (*this -> mlp_sig) -> forward(trk.to(torch::kFloat32)); 
    torch::Tensor rc_cmpr = torch::cat({hx_ij, hx_px, res_edge}, {-1}); 

    torch::Tensor rs_max  = std::get<0>(rs_cmpr.softmax(0).max({0}))*std::get<0>(rs_cmpr.max({0})); 
    torch::Tensor rhx_max = std::get<0>(rc_cmpr.softmax({0}).max({0}))*std::get<0>(rc_cmpr.max({0})); 

    torch::Tensor rs_min  = std::get<0>(rs_cmpr.softmax(0).min({0}))*std::get<0>(rs_cmpr.min({0}));
    torch::Tensor rhx_min = std::get<0>(rc_cmpr.softmax({0}).min({0}))*std::get<0>(rc_cmpr.min({0})); 

    torch::Tensor rs_mean  = rs_cmpr.softmax(0).mean({0})*rs_cmpr.mean({0}); 
    torch::Tensor rhx_mean = rc_cmpr.softmax({0}).mean({0})*rc_cmpr.mean({0}); 

    torch::Tensor _rx_hx = _ntops_hx * torch::cat({rhx_max.view({-1, 1}), rhx_min.view({-1, 1}), rhx_mean.view({-1, 1})}, {-1}); 
    torch::Tensor _rx_tp = _ntops_tp * torch::cat({rs_max.view({-1, 1}) , rs_min.view({-1, 1}) , rs_mean.view({-1, 1})}, {-1}); 
    torch::Tensor isres = _rx_tp.matmul(_rx_hx.transpose(1, 0)).view({-1, 2}).sum({0}, true).softmax(-1);

    this -> prediction_edge_feature("top_edge", top_edge); 
    this -> prediction_edge_feature("res_edge", res_edge); 
    this -> prediction_graph_feature("ntops", ntops);
    this -> prediction_graph_feature("signal", isres); 

}

experimental::~experimental(){}
model_template* experimental::clone(){return new experimental();}

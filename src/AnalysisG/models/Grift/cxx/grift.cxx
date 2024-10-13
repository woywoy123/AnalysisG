#include <grift.h>
#include <transform/cartesian-cuda.h>
#include <physics/cartesian-cuda.h>
#include <transform/polar-cuda.h>
#include <graph/graph-cuda.h>


grift::grift(){

    // create the null buffers
    this ->  x_nulls = torch::zeros({1, this -> _xrec}).to(torch::kInt); 
    this -> dx_nulls = torch::zeros({1, this -> _xrec}).to(torch::kFloat32);
    this -> te_nulls = torch::zeros({1, this -> _xout}).to(torch::kFloat32); 

    this -> rnn_x = new torch::nn::Sequential({
            {"rnn_x_l1", torch::nn::Linear(this -> _xin + this -> _xrec, this -> _hidden)},
            {"rnn_x_t1", torch::nn::Tanh()},
            {"rnn_x_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_x_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)}, 
            {"rnn_x_r2", torch::nn::ReLU()},
            {"rnn_x_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_x_t2", torch::nn::Tanh()},
            {"rnn_x_l3", torch::nn::Linear(this -> _hidden, this -> _xrec)}
    }); 

    int dxx_1 = this -> _xin*2 + this -> _xrec*4; 
    this -> rnn_dx = new torch::nn::Sequential({
            {"rnn_dx_l1", torch::nn::Linear(dxx_1, this -> _hidden)}, 
            {"rnn_dx_r1", torch::nn::ReLU()},
            {"rnn_dx_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_dx_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)},
            {"rnn_dx_s2", torch::nn::Tanh()},
            {"rnn_dx_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_dx_l3", torch::nn::Linear(this -> _hidden, this -> _xrec)}
    }); 

    this -> rnn_merge = new torch::nn::Sequential({
            {"rnn_mrg_l1", torch::nn::Linear(this -> _xin + this -> _xrec*2, this -> _hidden)}, 
            {"rnn_mrg_s1", torch::nn::ReLU()},
            {"rnn_mrg_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)},
            {"rnn_mrg_t2", torch::nn::Tanh()},
            {"rnn_mrg_l3", torch::nn::Linear(this -> _hidden, this -> _xrec)}
    }); 

    int dxx_2 = this -> _xrec*4 + this -> _xout; 
    this -> rnn_top_edge = new torch::nn::Sequential({
            {"rnn_top_l1", torch::nn::Linear(dxx_2, this -> _hidden)}, 
            {"rnn_top_r1", torch::nn::ReLU()},
            {"rnn_top_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_top_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)}, 
            {"rnn_top_t2", torch::nn::Tanh()},
            {"rnn_top_r2", torch::nn::ReLU()},
            {"rnn_top_l3", torch::nn::Linear(this -> _hidden, this -> _xout)},
            {"rnn_top_s3", torch::nn::Sigmoid()}
    }); 

    int dxx_r = this -> _xrec*3 + this -> _xout; 
    this -> rnn_res_edge = new torch::nn::Sequential({
            {"rnn_res_l1", torch::nn::Linear(dxx_r, this -> _hidden)}, 
            {"rnn_res_r1", torch::nn::ReLU()},
            {"rnn_res_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_res_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)}, 
            {"rnn_res_t2", torch::nn::Tanh()},
            {"rnn_res_r2", torch::nn::ReLU()},
            {"rnn_res_l3", torch::nn::Linear(this -> _hidden, this -> _xout)},
            {"rnn_res_s3", torch::nn::Sigmoid()}
    }); 

    this -> mlp_ntop = new torch::nn::Sequential({
            {"ntop_l1", torch::nn::Linear(this -> _xtop + this -> _xrec*2, this -> _xrec*2)}, 
            {"ntop_t1", torch::nn::Tanh()},
            {"ntop_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xrec*2}))}, 
            {"ntop_l2", torch::nn::Linear(this -> _xrec*2, this -> _xrec)}, 
            {"ntop_t2", torch::nn::Tanh()},
            {"ntop_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xrec}))}, 
            {"ntop_r3", torch::nn::ReLU()},
            {"ntop_l3", torch::nn::Linear(this -> _xrec, this -> _xtop)},
            {"ntop_s3", torch::nn::Sigmoid()}
    }); 

    this -> mlp_sig = new torch::nn::Sequential({
            {"res_l1", torch::nn::Linear(this -> _xout*2 + this -> _xtop + dxx_r, this -> _xrec*2)}, 
            {"res_t1", torch::nn::Tanh()},
            {"res_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xrec*2}))}, 
            {"res_l2", torch::nn::Linear(this -> _xrec*2, this -> _xrec)}, 
            {"res_t2", torch::nn::Tanh()},
            {"res_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xrec}))}, 
            {"res_r3", torch::nn::ReLU()},
            {"res_l3", torch::nn::Linear(this -> _xrec, this -> _xout)},
            {"res_s3", torch::nn::Sigmoid()}
    }); 

    this -> register_module(this -> rnn_x       );
    this -> register_module(this -> rnn_dx      );
    this -> register_module(this -> rnn_merge   );
    this -> register_module(this -> rnn_top_edge);
    this -> register_module(this -> rnn_res_edge);
    this -> register_module(this -> mlp_ntop    );
    this -> register_module(this -> mlp_sig     );


}

torch::Tensor grift::message(
        torch::Tensor _trk_i, torch::Tensor _trk_j, torch::Tensor pmc, torch::Tensor hx_i, torch::Tensor hx_j
){
    std::tuple<torch::Tensor, torch::Tensor> aggr; 
    torch::Tensor trk_ij = torch::cat({_trk_i, _trk_j}, {-1}); 

    aggr = graph::cuda::unique_aggregation(trk_ij, pmc); 
    torch::Tensor pmc_ij = std::get<0>(aggr); 
    torch::Tensor m_ij   = physics::cuda::cartesian::M(pmc_ij);
    torch::Tensor nds_ij = (std::get<1>(aggr) > -1).sum({-1}, true); 
    torch::Tensor fx_ij  = torch::cat({m_ij, pmc_ij, nds_ij, hx_i+hx_j}, {-1}).to(torch::kFloat32); 

    aggr = graph::cuda::unique_aggregation(_trk_i, pmc); 
    torch::Tensor pmc_i = std::get<0>(aggr); 
    torch::Tensor m_i   = physics::cuda::cartesian::M(pmc_i);
    torch::Tensor nds_i = (std::get<1>(aggr) > -1).sum({-1}, true); 
    torch::Tensor fx_i  = torch::cat({m_i, pmc_i, nds_i, hx_i}, {-1}).to(torch::kFloat32); 

    aggr = graph::cuda::unique_aggregation(_trk_j, pmc); 
    torch::Tensor pmc_j = std::get<0>(aggr); 
    torch::Tensor m_j   = physics::cuda::cartesian::M(pmc_i);
    torch::Tensor nds_j = (std::get<1>(aggr) > -1).sum({-1}, true); 
    torch::Tensor fx_j  = torch::cat({m_j, pmc_j, nds_j, hx_j}, {-1}).to(torch::kFloat32); 

    torch::Tensor dx = (*this -> rnn_dx) -> forward(torch::cat({fx_i, fx_j - fx_i, hx_i, hx_j - hx_i}, {-1})); 
    return (*this -> rnn_merge) -> forward(torch::cat({fx_ij, dx}, {-1})); 
}

void grift::forward(graph_t* data){

    // get the particle 4-vector and convert it to cartesian
    torch::Tensor* pt     = data -> get_data_node("pt", this);
    torch::Tensor* eta    = data -> get_data_node("eta", this);
    torch::Tensor* phi    = data -> get_data_node("phi", this);
    torch::Tensor* energy = data -> get_data_node("energy", this);
    torch::Tensor* is_lep = data -> get_data_node("is_lep", this); 
    torch::Tensor pmc     = transform::cuda::PxPyPzE(*pt, *eta, *phi, *energy) / 1000.0; 

    torch::Tensor edge_index = data -> get_edge_index(this) -> to(torch::kLong); 
    torch::Tensor src        = edge_index.index({0}).view({-1}); 
    torch::Tensor dst        = edge_index.index({1}).view({-1}); 

    // the event features
    torch::Tensor* num_jets = data -> get_data_graph("num_jets", this); 
    torch::Tensor* num_leps = data -> get_data_graph("num_leps", this); 
    torch::Tensor* met_phi  = data -> get_data_graph("phi", this);
    torch::Tensor* met      = data -> get_data_graph("met", this); 

    torch::Tensor num_bjet = data -> get_data_node("is_b", this) -> sum({0}, true);  
    torch::Tensor pid      = torch::cat({*num_jets, num_bjet, *num_leps, (*met)/1000.0, *met_phi}, {-1});  

    // ------ index the nodes from 0 to N-1 ----- //
    if (!this -> init){
        this -> x_nulls = this -> x_nulls.to(src.device()); 
        this -> dx_nulls = this -> dx_nulls.to(src.device()); 
        this -> te_nulls = this -> te_nulls.to(src.device()); 
        this -> init = true;
    }

    // ------ initialize nulls -------- //
    torch::Tensor trk = torch::zeros_like(*pt).to(torch::kInt); 
    torch::Tensor null_idx = torch::zeros_like(src); 
    torch::Tensor node_rnn = this -> x_nulls.index({trk.view({-1})}); 
    torch::Tensor edge_rnn = this -> dx_nulls.index({null_idx}); 
    torch::Tensor top_edge = this -> te_nulls.index({null_idx}); 
    torch::Tensor num_node = torch::ones_like(trk); 
    torch::Tensor node_i   = num_node.cumsum({0})-1; 

    // ------ index the edges from 0 to N^2 -1 ------ //
    unsigned int n_nodes = trk.size({0});
    torch::Tensor idx_mat = torch::zeros({n_nodes, n_nodes}, src.device()).to(torch::kLong); 
    idx_mat.index_put_({src, dst}, (null_idx+1).cumsum({-1})-1); 
   
    std::vector<torch::Tensor> gr_; 
    torch::Tensor top_edge_   = top_edge.clone(); 
    torch::Tensor edge_index_ = edge_index.clone();  

    while (edge_index_.size({1})){
        // ----- use the index matrix to map the source and destination edges to the edge index ----- //
        torch::Tensor src_ = edge_index_.index({0}); 
        torch::Tensor dst_ = edge_index_.index({1});
        torch::Tensor idx = idx_mat.index({src_, dst_}); 

        // ------------------ loop states ------------------------ //
        torch::Tensor hx_i  = node_rnn.index({src_});
        torch::Tensor hx_j  = node_rnn.index({dst_});  
        torch::Tensor hx_ij = edge_rnn; 

        // ------------------ create a new message --------------------- //
        edge_rnn = this -> message(node_i.index({src_}), node_i.index({dst_}), pmc, hx_i, hx_j); 
        hx_ij    = torch::cat({edge_rnn, hx_ij - edge_rnn, hx_i, hx_i - hx_j, top_edge_}, {-1}); 

        // ------------------ check edges for new paths ---------------- //
        top_edge_ = (*this -> rnn_top_edge) -> forward(hx_ij);

        // ----- update the top_edge prediction weights by index ------- //
        torch::Tensor sel = std::get<1>((top_edge_).max({-1})); 
        top_edge.index_put_({idx}, top_edge_ + top_edge.index({idx})); 

        // ---- check if the new prediction is simply null ---- /
        sel = sel < 1; 
        if (!sel.index({sel == false}).size({0})){break;}
        edge_index_ = edge_index_.index({torch::indexing::Slice(), sel}); 
        top_edge_   = top_edge_.index({sel}); 
        edge_rnn    = edge_rnn.index({sel}); 

        // ----------- create a new intermediate state of the nodes ----------- //
        gr_ = graph::cuda::edge_aggregation(edge_index, top_edge, pmc)["1"]; 
        node_i   = gr_[0].index({gr_[2]});
        num_node = (node_i > -1).sum({-1}, true); 
        node_rnn = torch::cat({physics::cuda::cartesian::M(gr_[3]), gr_[3], num_node, node_rnn}, {-1}); 
        node_rnn = (*this -> rnn_x) -> forward(node_rnn.to(torch::kFloat32)) / num_node;
    }

    // ----------- compress the top data ----------- //
    gr_ = graph::cuda::edge_aggregation(edge_index, top_edge, pmc)["1"]; 
    torch::Tensor node_trk = gr_[0].index({gr_[2]}); 
    num_node = (node_trk > -1).sum({-1}, true);
    torch::Tensor enc_tops = physics::cuda::cartesian::M(gr_[3]); 
    enc_tops = torch::cat({enc_tops, gr_[3], num_node, node_rnn}, {-1});
    enc_tops = (*this -> rnn_x) -> forward(enc_tops.to(torch::kFloat32)) / num_node; 

    torch::Tensor bkg = physics::cuda::cartesian::M(pmc); 
    bkg = torch::cat({bkg, pmc, torch::ones_like(trk), node_rnn}, {-1}); 
    bkg = (*this -> rnn_x) -> forward(bkg.to(torch::kFloat32));

    torch::Tensor ntops_ = torch::cat({enc_tops, bkg - enc_tops}, {-1});
    ntops_ = torch::cat({ntops_.sum({0}, true), pid}, {-1});
    ntops_ = (*this -> mlp_ntop) -> forward(ntops_.to(torch::kFloat32)); 

    torch::Tensor hxt_i = enc_tops.index({src}); 
    torch::Tensor hxt_j = enc_tops.index({dst}); 
    torch::Tensor res_rnn = this -> message(node_trk.index({src}), node_trk.index({dst}), pmc, hxt_i, hxt_j); 
    torch::Tensor fx_ij  = torch::cat({res_rnn, top_edge, hxt_i, hxt_j - hxt_i}, {-1});
    torch::Tensor res_edge = (*this -> rnn_res_edge) -> forward(fx_ij);

    torch::Tensor isres_ = torch::cat({res_edge, top_edge - res_edge, ntops_.index({null_idx}), fx_ij}, {-1}); 
    isres_ = (*this -> mlp_sig) -> forward(isres_).sum({0}, true) / res_edge.size({0});
    
    this -> prediction_edge_feature("top_edge", top_edge); 
    this -> prediction_edge_feature("res_edge", res_edge); 

    this -> prediction_graph_feature("ntops", ntops_);
    this -> prediction_graph_feature("signal", isres_); 
    if (!this -> inference_mode){return;}

    this -> prediction_extra("top_edge_score", top_edge.softmax(-1));
    this -> prediction_extra("res_edge_score", res_edge.softmax(-1));
    this -> prediction_extra("ntops_score"   , ntops_.softmax(-1).view({-1})); 
    this -> prediction_extra("is_res_score"  , isres_.softmax(-1).view({-1})); 

    this -> prediction_extra("is_lep"        , is_lep -> view({-1})); 
    this -> prediction_extra("num_leps"      , num_leps -> view({-1})); 
    this -> prediction_extra("num_jets"      , num_jets -> view({-1})); 
    this -> prediction_extra("num_bjets"     , num_bjet.view({-1})); 
    if (!this -> is_mc){return;}

    torch::Tensor ntops_t  = data -> get_truth_graph("ntops"  , this) -> view({-1}); 
    torch::Tensor signa_t  = data -> get_truth_graph("signal" , this) -> view({-1});
    torch::Tensor r_edge_t = data -> get_truth_edge("res_edge", this) -> view({-1}); 
    torch::Tensor t_edge_t = data -> get_truth_edge("top_edge", this) -> view({-1}); 

    this -> prediction_extra("truth_ntops"   , ntops_t); 
    this -> prediction_extra("truth_signal"  , signa_t); 
    this -> prediction_extra("truth_res_edge", r_edge_t); 
    this -> prediction_extra("truth_top_edge", t_edge_t); 
}

grift::~grift(){}
model_template* grift::clone(){
    grift* md = new grift(); 
    md -> drop_out = this -> drop_out; 
    md -> is_mc    = this -> is_mc; 
    return md; 
}
#include <grift.h>
#include <pyc/pyc.h>

grift::grift(){

    // create the null buffers
    this ->  x_nulls = torch::zeros({1, this -> _xrec}).to(torch::kInt); 
    this -> dx_nulls = torch::zeros({1, this -> _xrec}).to(torch::kFloat32);
    this -> te_nulls = torch::zeros({1, this -> _xout}).to(torch::kFloat32); 

    this -> rnn_x = new torch::nn::Sequential({
            {"rnn_x_l1", torch::nn::Linear(this -> _xin + this -> _xrec, this -> _xrec + this -> _xin)},
            {"rnn_x_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xrec + this -> _xin}))}, 
            {"rnn_x_r1", torch::nn::LeakyReLU()},
            {"rnn_x_l2", torch::nn::Linear(this -> _xrec + this -> _xin, this -> _xrec)}, 
            {"rnn_x_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xrec}))}, 
            {"rnn_x_t2", torch::nn::Tanh()},
            {"rnn_x_l3", torch::nn::Linear(this -> _xrec, this -> _xrec)}
    }); 

    this -> rnn_dx = new torch::nn::Sequential({
            {"rnn_dx_l1", torch::nn::Linear(this -> _xrec * 3, this -> _xrec * 3)}, 
            {"rnn_dx_r1", torch::nn::LeakyReLU()},
            {"rnn_dx_t1", torch::nn::Tanh()},
            {"rnn_dx_l2", torch::nn::Linear(this -> _xrec * 3, this -> _xrec * 3)}, 
            {"rnn_dx_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xrec * 3}))}, 
            {"rnn_dx_t2", torch::nn::Tanh()},
            {"rnn_dx_l3", torch::nn::Linear(this -> _xrec * 3, this -> _xrec)}
    }); 

    this -> rnn_hxx = new torch::nn::Sequential({
            {"rnn_hxx_l1", torch::nn::Linear(this -> _xrec*3 + 2, this -> _xrec*3 + 2)}, 
            {"rnn_hxx_r1", torch::nn::LeakyReLU()},
            {"rnn_hxx_t1", torch::nn::Tanh()},
            {"rnn_hxx_l2", torch::nn::Linear(this -> _xrec*3 + 2, this -> _xrec * 2)}, 
            {"rnn_hxx_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xrec * 2}))}, 
            {"rnn_hxx_r2", torch::nn::LeakyReLU()},
            {"rnn_hxx_t2", torch::nn::Sigmoid()},
            {"rnn_hxx_l3", torch::nn::Linear(this -> _xrec * 2, this -> _xrec)}
    }); 

    this -> rnn_txx = new torch::nn::Sequential({
            {"rnn_txx_l1", torch::nn::Linear(this -> _xrec * 2, this -> _xrec * 2)}, 
            {"rnn_txx_r1", torch::nn::ReLU()},
            {"rnn_txx_l2", torch::nn::Linear(this -> _xrec * 2, this -> _xrec)}, 
            {"rnn_txx_t2", torch::nn::Sigmoid()},
            {"rnn_txx_l3", torch::nn::Linear(this -> _xrec, this -> _xout)}
    }); 

    int dxx_r = this -> _xrec*4; 
    this -> rnn_rxx = new torch::nn::Sequential({
            {"rnn_rxx_l1", torch::nn::Linear(dxx_r, this -> _hidden)}, 
            {"rnn_rxx_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_rxx_r1", torch::nn::LeakyReLU()},
            {"rnn_rxx_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)}, 
            {"rnn_rxx_t2", torch::nn::Sigmoid()},
            {"rnn_rxx_l3", torch::nn::Linear(this -> _hidden, this -> _xout)}
    }); 

    this -> mlp_ntop = new torch::nn::Sequential({
            {"ntop_l1", torch::nn::Linear(this -> _xtop + this -> _xrec, this -> _xrec)}, 
            {"ntop_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xrec}))}, 
            {"ntop_r1", torch::nn::LeakyReLU()},
            {"ntop_l2", torch::nn::Linear(this -> _xrec, this -> _xrec)}, 
            {"ntop_t2", torch::nn::Sigmoid()},
            {"ntop_l3", torch::nn::Linear(this -> _xrec, this -> _xtop)}
    }); 

    this -> mlp_sig = new torch::nn::Sequential({
            {"res_l1", torch::nn::Linear(this -> _xout*2 + dxx_r + this -> _xtop*2, this -> _xrec*2)}, 
            {"res_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xrec*2}))}, 
            {"res_r1", torch::nn::LeakyReLU()},
            {"res_l2", torch::nn::Linear(this -> _xrec*2, this -> _xrec)}, 
            {"res_t2", torch::nn::Sigmoid()},
            {"res_l3", torch::nn::Linear(this -> _xrec, this -> _xout)}
    }); 

    this -> register_module(this -> rnn_x   ); //, mlp_init::xavier_uniform);
    this -> register_module(this -> rnn_dx  ); //, mlp_init::xavier_uniform);
    this -> register_module(this -> rnn_hxx ); //, mlp_init::xavier_uniform); 
    this -> register_module(this -> rnn_txx ); //, mlp_init::xavier_uniform);
    this -> register_module(this -> rnn_rxx ); //, mlp_init::xavier_uniform);
    this -> register_module(this -> mlp_ntop); //, mlp_init::xavier_uniform);
    this -> register_module(this -> mlp_sig ); //, mlp_init::xavier_uniform);


}

torch::Tensor grift::node_encode(torch::Tensor pmc, torch::Tensor num_node, torch::Tensor* node_rnn){
    num_node = (num_node > -1).sum({-1}, true); 
    torch::Tensor mass = torch::abs(pyc::physics::cartesian::combined::M(pmc)); 
    if (!node_rnn){return torch::cat({mass, pmc, num_node}, {-1}).to(torch::kFloat32);}
    torch::Tensor nox = torch::cat({mass, pmc, num_node, *node_rnn}, {-1});
    return (*this -> rnn_x) -> forward(nox.to(torch::kFloat32)) * 1.0 / num_node.to(torch::kFloat32);  
}

torch::Tensor grift::message(
        torch::Tensor _trk_i, torch::Tensor _trk_j, torch::Tensor pmc, torch::Tensor hx_i, torch::Tensor hx_j
){
    const std::string key_idx = "unique"; 
    const std::string key_smx = "node-sum"; 

    torch::Dict<std::string, torch::Tensor> aggr; 
    aggr = pyc::graph::unique_aggregation(torch::cat({_trk_i, _trk_j}, {-1}), pmc); 
    torch::Tensor fx_ij = this -> node_encode(aggr.at(key_smx), aggr.at(key_idx), &hx_i); 

    aggr = pyc::graph::unique_aggregation(_trk_i, pmc); 
    torch::Tensor fx_i = this -> node_encode(aggr.at(key_smx), aggr.at(key_idx), &hx_i); 

    aggr = pyc::graph::unique_aggregation(_trk_j, pmc); 
    torch::Tensor fx_j = this -> node_encode(aggr.at(key_smx), aggr.at(key_idx), &hx_j); 

    return (*this -> rnn_dx) -> forward(torch::cat({fx_ij, fx_i, torch::abs(fx_j - fx_i)}, {-1})); 
}


torch::Tensor grift::recurse(
        torch::Tensor* node_i,       torch::Tensor* idx_mat,  
        torch::Tensor* edge_index_,  torch::Tensor* edge_index, 
        torch::Tensor* edge_rnn,     torch::Tensor* node_dnn,
        torch::Tensor* top_edge,     torch::Tensor* pmc,
        torch::Tensor* node_s
){

    const std::string key_smx = "cls::1::node-sum"; 
    const std::string key_idx = "cls::1::node-indices"; 
    torch::Tensor hx_ij; 

    // ----- use the index matrix to map the source and destination edges to the edge index ----- //
    torch::Tensor src_ = edge_index_ -> index({0}); 
    torch::Tensor dst_ = edge_index_ -> index({1});
    torch::Tensor idx  = idx_mat -> index({src_, dst_}); 

    torch::Tensor hx_i = node_dnn -> index({src_});
    torch::Tensor hx_j = node_dnn -> index({dst_});
    
    torch::Tensor nx_i = node_i -> index({src_}); 
    torch::Tensor nx_j = node_i -> index({dst_}); 
    torch::Tensor r_dx = edge_rnn -> index({idx}); 
    
    // ------------------ create a new message --------------------- //
    hx_ij = this -> message(nx_i, nx_j, *pmc, hx_i, hx_j); 

    // ------------------ check edges for new state transititons --------------- //
    torch::Tensor top_idx = (*this -> rnn_txx) -> forward(torch::cat({hx_ij, r_dx - hx_ij}, {-1})); 
    top_edge -> index_put_({idx}, top_idx); 

    // ----------- create a new intermediate state of the nodes ----------- //
    torch::Dict<std::string, torch::Tensor> gr_ = pyc::graph::edge_aggregation(*edge_index, *top_edge, *pmc); 
    torch::Tensor hk_i = this -> node_encode(gr_.at(key_smx), gr_.at(key_idx), node_dnn); 

    // protects against batching
    torch::Tensor skxi = ((*node_i) > -1).sum({-1}) != ((gr_.at(key_idx) > -1).sum({-1})); 
    node_dnn -> index_put_({skxi}, hk_i.index({skxi})); 
    if (skxi.index({skxi}).size({0})){
        torch::Tensor sel = std::get<1>(top_edge -> max({-1})) > 0; 
        return edge_index -> index({torch::indexing::Slice(), sel == false});
    }

    (*node_i) = gr_.at(key_idx); 
    torch::Tensor srcx = (torch::ones_like((*node_i)) * (*node_s)).view({1, -1});
    torch::Tensor dstx = node_i -> reshape({1, -1}); 
    torch::Tensor msk_ = dstx.reshape({-1}) > -1;

    // update the intermediary recursion state from i -> j'
    hx_ij = torch::cat({node_dnn -> index({src_}), hx_ij, node_dnn -> index({dst_}) - node_dnn -> index({src_}), top_idx}, {-1});
    hx_ij = (*this -> rnn_hxx) -> forward(hx_ij); 
 
    // ------ walk to the next node (nxt) ------- //
    edge_rnn -> index_put_({idx}, hx_ij); 
    return torch::cat({
            torch::cat({srcx}, {-1}), 
            torch::cat({dstx}, {-1})
    }, {0}).index({torch::indexing::Slice(), msk_}); 
}


void grift::forward(graph_t* data){

    // get the particle 4-vector and convert it to cartesian
    torch::Tensor batch_index  = data -> get_batch_index(this) -> view({-1}).clone(); 
    torch::Tensor event_index  = std::get<0>(torch::_unique(batch_index)); 
    torch::Tensor* pt          = data -> get_data_node("pt", this);
    torch::Tensor* eta         = data -> get_data_node("eta", this);
    torch::Tensor* phi         = data -> get_data_node("phi", this);
    torch::Tensor* energy      = data -> get_data_node("energy", this);
    torch::Tensor* is_lep      = data -> get_data_node("is_lep", this); 
    torch::Tensor pmc          = pyc::transform::separate::PxPyPzE(*pt, *eta, *phi, *energy) / 1000.0; 

    torch::Tensor edge_index   = data -> get_edge_index(this) -> to(torch::kLong); 
    torch::Tensor src          = edge_index.index({0}).view({-1}); 
    torch::Tensor dst          = edge_index.index({1}).view({-1}); 

    // the event features
    torch::Tensor* num_jets = data -> get_data_graph("num_jets", this); 
    torch::Tensor* num_leps = data -> get_data_graph("num_leps", this); 
    torch::Tensor* met_phi  = data -> get_data_graph("phi", this);
    torch::Tensor* met      = data -> get_data_graph("met", this); 

    torch::Tensor num_bjet   = data -> get_data_node("is_b", this) -> clone(); 
    torch::Tensor num_bjets_ = torch::zeros({event_index.size({0}), 1}, num_bjet.device()).to(num_bjet.dtype()); 
    num_bjets_.index_add_({0}, batch_index, num_bjet); 
    torch::Tensor pid = torch::cat({*num_jets, num_bjets_, *num_leps, (*met)/1000.0, *met_phi}, {-1});  

    // ------ index the nodes from 0 to N-1 ----- //
    if (!this -> init){
        this -> x_nulls  = this -> x_nulls.to(src.device()); 
        this -> dx_nulls = this -> dx_nulls.to(src.device()); 
        this -> te_nulls = this -> te_nulls.to(src.device()); 
        this -> init = true;
    }

    // ------ initialize nulls -------- //
    torch::Tensor trk      = torch::zeros_like(*pt).to(torch::kInt); 
    torch::Tensor null_idx = torch::zeros_like(src); 
    torch::Tensor node_rnn = this -> x_nulls.index({trk.view({-1})}).to(torch::kFloat32); 
    torch::Tensor edge_rnn = torch::zeros_like(this -> dx_nulls.index({null_idx})).softmax(-1); 
    torch::Tensor top_edge = torch::zeros_like(this -> te_nulls.index({null_idx})).softmax(-1); 
    torch::Tensor num_node = torch::ones_like(trk); 
    torch::Tensor node_i   = num_node.cumsum({0})-1;
    torch::Tensor node_s   = node_i.clone(); 

    node_rnn = this -> node_encode(pmc, num_node, &node_rnn); 

    // ------ index the edges from 0 to N^2 -1 ------ //
    unsigned int n_nodes  = data -> num_nodes;
    torch::Tensor idx_mat = torch::zeros({n_nodes, n_nodes}, src.device()).to(torch::kLong);  
    idx_mat.index_put_({src, dst}, (null_idx+1).cumsum({-1})-1); 

    const std::string key_idx = "cls::1::node-indices"; 
    const std::string key_smx = "cls::1::node-sum"; 

    long numx = edge_index.size({1}); 
    torch::Tensor edge_index_ = edge_index.clone();  
    while (edge_index_.size({1})){
        torch::Tensor p_index_ = this -> recurse(
            &node_i, &idx_mat,  &edge_index_, &edge_index, 
            &edge_rnn, &node_rnn, &top_edge, &pmc, &node_s
        ); 

        long nump = p_index_.size({1}); 
        if (!nump){break;}
        if (numx != nump){numx = nump; edge_index_ = p_index_; continue;}
        torch::Tensor idx = edge_index_.view({-1}) == p_index_.view({-1}); 
        if (idx.index({idx}).size({0}) == nump*2){break;}
        numx = nump; edge_index_ = p_index_;
    }

    // ----------- compress the top data ----------- //
    torch::Dict<std::string, torch::Tensor> gr_ = pyc::graph::edge_aggregation(edge_index, top_edge, pmc); 
    torch::Tensor node_trk = gr_.at(key_idx); 
    torch::Tensor ntops = this -> node_encode(gr_.at(key_smx), node_trk, &node_rnn); 
    torch::Tensor tmlp  = torch::zeros({event_index.size({0}), ntops.size({1})}, ntops.device()).to(ntops.dtype()); 
    tmlp.index_add_({0}, batch_index, ntops); 
    tmlp = torch::cat({tmlp, pid}, {-1}); 
    tmlp = (*this -> mlp_ntop) -> forward(tmlp.to(torch::kFloat32));

    torch::Tensor hxt_i = node_rnn.index({src}); 
    torch::Tensor hxt_j = node_rnn.index({dst}); 

    gr_ = pyc::graph::unique_aggregation(torch::cat({node_trk.index({src}), node_trk.index({dst})}, {-1}), pmc); 
    torch::Tensor node_res = this -> node_encode(gr_.at("node-sum"), gr_.at("unique"), &hxt_i); 
    torch::Tensor fx_ij    = torch::cat({node_res, ntops.index({src}), hxt_i, hxt_j - hxt_i}, {-1});
    torch::Tensor res_edge = (*this -> rnn_rxx) -> forward(fx_ij);

    torch::Tensor isres_ = torch::cat({res_edge, top_edge - res_edge, fx_ij}, {-1}); 
    torch::Tensor tmp = torch::zeros({event_index.size({0}), isres_.size({1})}, isres_.device()).to(isres_.dtype()); 
    tmp.index_add_({0}, batch_index.index({src}), isres_); 

    isres_ = torch::cat({tmp, pid, tmlp}, {-1}); 
    isres_ = (*this -> mlp_sig) -> forward(isres_.to(torch::kFloat32));
   
    this -> prediction_edge_feature("top_edge", top_edge); 
    this -> prediction_edge_feature("res_edge", res_edge); 

    this -> prediction_graph_feature("ntops" , tmlp);
    this -> prediction_graph_feature("signal", isres_); 

    if (!this -> inference_mode){return;}
    if (this -> pagerank){
        gr_ = pyc::graph::PageRankReconstruction(edge_index, top_edge, pmc); 
        this -> prediction_extra("page-nodes", gr_.at("page-nodes")); 
        this -> prediction_extra("page-mass" , gr_.at("page-mass" )); 
    }

    this -> prediction_extra("top_edge_score", top_edge.softmax(-1));
    this -> prediction_extra("res_edge_score", res_edge.softmax(-1));
    this -> prediction_extra("ntops_score"   , tmlp.softmax(-1)); 
    this -> prediction_extra("is_res_score"  , isres_.softmax(-1)); 

    this -> prediction_extra("is_lep"        , *is_lep); 
    this -> prediction_extra("num_leps"      , *num_leps); 
    this -> prediction_extra("num_jets"      , *num_jets); 
    this -> prediction_extra("num_bjets"     , num_bjets_); 
    if (!this -> is_mc){return;}

    torch::Tensor* ntops_t  = data -> get_truth_graph("ntops"  , this); 
    torch::Tensor* signa_t  = data -> get_truth_graph("signal" , this);
    torch::Tensor* r_edge_t = data -> get_truth_edge("res_edge", this); 
    torch::Tensor* t_edge_t = data -> get_truth_edge("top_edge", this); 

    this -> prediction_extra("truth_ntops"   , *ntops_t); 
    this -> prediction_extra("truth_signal"  , *signa_t); 
    this -> prediction_extra("truth_res_edge", *r_edge_t); 
    this -> prediction_extra("truth_top_edge", *t_edge_t); 
}

grift::~grift(){}
model_template* grift::clone(){
    grift* md = new grift(); 
    md -> is_mc    = this -> is_mc; 
    md -> pagerank = this -> pagerank;
    return md; 
}

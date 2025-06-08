#include <grift.h>
#include <pyc/pyc.h>

grift::grift(){
    // create the null buffers
    this ->  x_nulls = torch::zeros({1, this -> _xrec}).to(torch::kInt); 
    this -> dx_nulls = torch::zeros({1, this -> _xrec}).to(torch::kFloat32);
    this -> te_nulls = torch::zeros({1, this -> _xout}).to(torch::kFloat32); 
    this -> retain_graph = false;
    this -> enable_anomaly = false;  

    this -> rnn_x = new torch::nn::Sequential({
            {"x_l1", torch::nn::Linear(this -> _xin + this -> _xrec, this -> _hidden)},
            {"x_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"x_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)}, 
            {"x_r2", torch::nn::LeakyReLU()},
            {"x_t2", torch::nn::Tanh()},
            {"x_l3", torch::nn::Linear(this -> _hidden, this -> _xrec)},
            {"x_n3", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xrec}))}, 
            {"x_l4", torch::nn::Linear(this -> _xrec, this -> _xrec)}
    }); 

    this -> rnn_dx = new torch::nn::Sequential({
            {"dx_l1", torch::nn::Linear(this -> _xrec*2 + this -> _xin, this -> _xrec*2)}, 
            {"dx_t1", torch::nn::ReLU()},
            {"dx_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xrec*2}))}, 
            {"dx_l2", torch::nn::Linear(this -> _xrec*2, this -> _xrec)}, 
            {"dx_r2", torch::nn::Sigmoid()},
            {"dx_l3", torch::nn::Linear(this -> _xrec, this -> _xrec)}
    }); 

    this -> rnn_hxx = new torch::nn::Sequential({
            {"hxx_l1", torch::nn::Linear(this -> _xrec*2, this -> _xrec*2)}, 
            {"hxx_t1", torch::nn::Tanh()},
            {"hxx_l2", torch::nn::Linear(this -> _xrec*2, this -> _xrec)}, 
            {"hxx_r1", torch::nn::ReLU()},
            {"hxx_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xrec}))}, 
            {"hxx_l3", torch::nn::Linear(this -> _xrec, this -> _xrec)}
    }); 

    this -> rnn_txx = new torch::nn::Sequential({
            {"top_l1", torch::nn::Linear(this -> _xrec, this -> _xrec)}, 
            {"top_r1", torch::nn::ReLU()},
            {"top_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xrec}))}, 
            {"top_t1", torch::nn::Tanh()},
            {"top_l2", torch::nn::Linear(this -> _xrec, this -> _xout)}
    }); 

    this -> rnn_rxx = new torch::nn::Sequential({
            {"res_l1", torch::nn::Linear(this -> _xrec*4, this -> _xrec*4)}, 
            {"res_r1", torch::nn::ReLU()},
            {"res_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xrec*4}))}, 
            {"res_t1", torch::nn::Tanh()},
            {"res_l2", torch::nn::Linear(this -> _xrec*4, this -> _xout)}
    }); 

    this -> mlp_ntop = new torch::nn::Sequential({
            {"ntop_l1", torch::nn::Linear(this -> _xtop + this -> _xrec, this -> _xrec)}, 
            {"ntop_t1", torch::nn::Tanh()},
            {"ntop_l2", torch::nn::Linear(this -> _xrec, this -> _xrec)}, 
            {"ntop_r2", torch::nn::ReLU()},
            {"ntop_m2", torch::nn::AdaptiveAvgPool1d(torch::nn::AdaptiveAvgPool1dOptions(this -> _xtop))},
            {"ntop_t2", torch::nn::Tanh()},
            {"ntop_l3", torch::nn::Linear(this -> _xtop, this -> _xtop)}
    }); 

    int dxx_r = this -> _xrec*4 + this -> _xout + this -> _xtop*2; 
    this -> mlp_sig = new torch::nn::Sequential({
            {"sig_l1", torch::nn::Linear(dxx_r, this -> _hidden)}, 
            {"sig_r1", torch::nn::LeakyReLU()},
            {"sig_l2", torch::nn::Linear(this -> _hidden, this -> _xrec)}, 
            {"sig_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xrec}))}, 
            {"sig_l3", torch::nn::Linear(this -> _xrec, this -> _xout)}, 
            {"sig_t3", torch::nn::Tanh()},
            {"sig_l4", torch::nn::Linear(this -> _xout, this -> _xout)}

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
    if (!node_rnn){return torch::cat({pyc::physics::cartesian::combined::M(pmc), pmc, num_node}, {-1}).to(torch::kFloat32);}
    torch::Tensor nox = torch::cat({pyc::physics::cartesian::combined::M(pmc), pmc, num_node, *node_rnn}, {-1}); 
    return (*this -> rnn_x) -> forward(nox.to(torch::kFloat32));  
}


torch::Tensor grift::message(
        torch::Tensor _trk_i, torch::Tensor _trk_j, torch::Tensor pmc, torch::Tensor* hx_i, torch::Tensor* hx_j
){
    const std::string key_idx = "unique"; 
    const std::string key_smx = "node-sum"; 
    torch::Dict<std::string, torch::Tensor> aggr; 

    aggr = pyc::graph::unique_aggregation(torch::cat({_trk_i, _trk_j}, {-1}), pmc); 
    torch::Tensor fx_ij = this -> node_encode(aggr.at(key_smx), (aggr.at(key_idx) > -1).sum({-1}, true), nullptr);   

    aggr = pyc::graph::unique_aggregation(_trk_i, pmc); 
    torch::Tensor fx_i  = this -> node_encode(aggr.at(key_smx), (aggr.at(key_idx) > -1).sum({-1}, true), hx_i);   

    aggr = pyc::graph::unique_aggregation(_trk_j, pmc); 
    torch::Tensor fx_j  = this -> node_encode(aggr.at(key_smx), (aggr.at(key_idx) > -1).sum({-1}, true), hx_j);   

    return (*this -> rnn_dx) -> forward(torch::cat({fx_ij, fx_i, fx_j - fx_i}, {-1})); 
}

void grift::forward(graph_t* data){

    // get the particle 4-vector and convert it to cartesian
    torch::Tensor batch_index  = data -> get_batch_index(this) -> view({-1}).clone(); 
    torch::Tensor event_index  = std::get<0>(torch::_unique(batch_index)); 
    torch::Tensor* pt          = data -> get_data_node("pt"    , this);
    torch::Tensor* eta         = data -> get_data_node("eta"   , this);
    torch::Tensor* phi         = data -> get_data_node("phi"   , this);
    torch::Tensor* energy      = data -> get_data_node("energy", this);
    torch::Tensor* is_lep      = data -> get_data_node("is_lep", this); 
    torch::Tensor  pmc         = pyc::transform::combined::PxPyPzE(torch::cat({*pt, *eta, *phi, *energy}, {-1})) / 1000.0; 

    torch::Tensor edge_index   = data -> get_edge_index(this) -> to(torch::kLong); 
    torch::Tensor src          = edge_index.index({0}).view({-1}); 
    torch::Tensor dst          = edge_index.index({1}).view({-1}); 

    // the event features
    torch::Tensor* num_jets = data -> get_data_graph("num_jets", this); 
    torch::Tensor* num_leps = data -> get_data_graph("num_leps", this); 
    torch::Tensor* met_phi  = data -> get_data_graph("phi", this);
    torch::Tensor* met      = data -> get_data_graph("met", this); 

    torch::Tensor num_bjet = data -> get_data_node("is_b", this) -> clone(); 
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
    torch::Tensor trk = torch::zeros_like(*pt).to(torch::kInt); 
    torch::Tensor null_idx = torch::zeros_like(src); 
    torch::Tensor node_rnn = torch::zeros_like(this -> x_nulls.index({trk.view({-1})}).to(torch::kFloat32)); 
    torch::Tensor edge_rnn = torch::zeros_like(this -> dx_nulls.index({null_idx})); 
    torch::Tensor top_edge = torch::zeros_like(this -> te_nulls.index({null_idx})); 
    torch::Tensor num_node = torch::ones_like(trk); 
    torch::Tensor node_i_  = num_node.cumsum({0})-1;  

    // ------ index the edges from 0 to N^2 -1 ------ //
    unsigned int n_nodes  = data -> num_nodes;
    torch::Tensor idx_mat = torch::zeros({n_nodes, n_nodes}, src.device()).to(torch::kLong);  
    idx_mat.index_put_({src, dst}, (null_idx+1).cumsum({-1})-1); 
    torch::Tensor norm = torch::zeros_like(idx_mat); 

    const std::string key_idx = "cls::1::node-indices"; 
    const std::string key_smx = "cls::1::node-sum"; 

    torch::Tensor edge_index_ = edge_index;  
    while (edge_index_.size({1})){
        // ----- use the index matrix to map the source and destination edges to the edge index ----- //
        torch::Tensor src_  = edge_index_.index({0}).view({-1}); 
        torch::Tensor dst_  = edge_index_.index({1}).view({-1}); 
        torch::Tensor idx   = idx_mat.index({src_, dst_}); 
        torch::Tensor hx_i  = node_rnn.index({src_}); 
        torch::Tensor hx_j  = node_rnn.index({dst_});

        // ------------------ loop states ------------------------ //
        // ------------------ create a new message --------------------- //
        torch::Tensor dx_ij = this -> message(node_i_.index({src_}), node_i_.index({dst_}), pmc, &hx_i, &hx_j); 

        // ----- update the top_edge prediction weights by index ------- //
        edge_rnn.index_put_({idx}, dx_ij); 
        top_edge = (*this -> rnn_txx) -> forward(edge_rnn);

        // ---- check if the new prediction is simply null ---- /
        torch::Tensor msk = (std::get<1>(top_edge.max({-1})) < 1).index({idx}); 

        // ------ protection against depleted event graphs ---------- //
        torch::Tensor skp = norm.sum({-1}, true); 
        torch::Tensor msx = (norm.index({src_, dst_}) + msk) > 0; 
        norm.index_put_({src_, dst_}, msx*1); 
        skp = (norm.sum({-1}, true) > skp).view({-1});  

        // ----------- create a new intermediate state of the nodes ----------- //
        torch::Dict<std::string, torch::Tensor> gr_ = pyc::graph::edge_aggregation(edge_index, top_edge, pmc); 
        torch::Tensor nodes = (gr_.at(key_idx) > -1).sum({-1}, true); 
        hx_j = this -> node_encode(gr_.at(key_smx), nodes, &node_rnn).index({skp}); 
        node_rnn.index_put_({skp}, hx_j);
        
        nodes = nodes.index({src})*nodes.index({dst}); 
        nodes = 1.0 / torch::pow(nodes, 0.5); 

        hx_i  = torch::zeros_like(node_rnn); 
        hx_i.index_add_({0}, src_.index({msk == false}), dx_ij.index({msk == false})); 

        edge_rnn = (*this -> rnn_hxx) -> forward(torch::cat({hx_i.index({src}), node_rnn.index({dst}) - node_rnn.index({src})}, {-1}))*nodes;
        if (!skp.index({skp}).size({0})){break;}
        if (!msk.index({msk == false}).size({0})){break;}

        // ------ walk to the next node (nxt) ------- //
        edge_index_ = edge_index_.index({torch::indexing::Slice(), msk});  
        node_i_ = gr_.at(key_idx); 
    }

    // ----------- compress the top data ----------- //
    torch::Dict<std::string, torch::Tensor> gr_ = pyc::graph::edge_aggregation(edge_index, top_edge, pmc); 
    trk = gr_.at(key_idx); 
    torch::Tensor ntops = this -> node_encode(gr_.at(key_smx), (trk > -1).sum({-1}, true), &node_rnn); 
    torch::Tensor tmlp  = torch::zeros({event_index.size({0}), ntops.size({1})}, ntops.device()).to(ntops.dtype()); 
    tmlp.index_add_({0}, batch_index, ntops); 
    tmlp = torch::cat({tmlp, pid}, {-1}).to(torch::kFloat32); 
    tmlp = (*this -> mlp_ntop) -> forward(tmlp);

    torch::Tensor hx_i = node_rnn.index({src}); 
    gr_  = pyc::graph::unique_aggregation(torch::cat({trk.index({src}), trk.index({dst})}, {-1}), pmc); 
    torch::Tensor node_res  = this -> node_encode(gr_.at("node-sum"), (gr_.at("unique") > -1).sum({-1}, true), &hx_i); 

    torch::Tensor fx_ij = torch::cat({node_res, ntops.index({src}), hx_i, edge_rnn}, {-1});
    torch::Tensor res_edge = (*this -> rnn_rxx) -> forward(fx_ij)*top_edge.softmax(-1);

    torch::Tensor isres_ = torch::cat({fx_ij, res_edge}, {-1}); 
    torch::Tensor tmp = torch::zeros({event_index.size({0}), isres_.size({1})}, isres_.device()).to(isres_.dtype()); 
    tmp.index_add_({0}, batch_index.index({src}), isres_); 
    isres_ = torch::cat({tmp, pid, tmlp}, {-1}); 
    isres_ = (*this -> mlp_sig) -> forward(isres_.to(torch::kFloat32));
  
    this -> prediction_edge_feature("top_edge", top_edge); 
    this -> prediction_edge_feature("res_edge", res_edge); 
    this -> prediction_graph_feature("ntops"  , tmlp    );
    this -> prediction_graph_feature("signal" , isres_  );

    if (!this -> inference_mode){return;}
    if (this -> pagerank){
        gr_ = pyc::graph::PageRankReconstruction(edge_index, top_edge, pmc);
        this -> prediction_extra("page-nodes", gr_.at("page-nodes")); 
        this -> prediction_extra("page-mass" , gr_.at("page-mass")); 
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
    md -> drop_out = this -> drop_out; 
    md -> is_mc    = this -> is_mc; 
    md -> pagerank = this -> pagerank; 
    return md; 
}

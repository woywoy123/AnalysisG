#include <grift.h>
#include <pyc/pyc.h>

grift::grift(){
    // create the null buffers
    this ->  x_nulls = torch::zeros({1, this -> _xrec}).to(torch::kInt); 
    this -> dx_nulls = torch::zeros({1, this -> _xrec}).to(torch::kFloat32);
    this -> te_nulls = torch::zeros({1, this -> _xout}).to(torch::kFloat32); 

    this -> rnn_x = new torch::nn::Sequential({
            {"rnn_x_l1", torch::nn::Linear(this -> _xin + this -> _xrec, this -> _hidden)},
            {"rnn_x_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_x_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)}, 
            {"rnn_x_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_x_l3", torch::nn::Linear(this -> _hidden, this -> _xrec)}
    }); 

    int dxx_1 = (this -> _xin + this -> _xrec)*3; 
    this -> rnn_dx = new torch::nn::Sequential({
            {"rnn_dx_l1", torch::nn::Linear(dxx_1, this -> _hidden)}, 
            {"rnn_dx_r1", torch::nn::ReLU()},
            {"rnn_dx_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_dx_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)}, 
            {"rnn_dx_s2", torch::nn::Sigmoid()},
            {"rnn_dx_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_dx_l3", torch::nn::Linear(this -> _hidden, this -> _xrec)}
    }); 

    this -> rnn_hxx = new torch::nn::Sequential({
            {"rnn_hxx_l1", torch::nn::Linear(this -> _xrec*4 + this -> _xout, this -> _hidden)}, 
            {"rnn_hxx_s1", torch::nn::Sigmoid()},
            {"rnn_hxx_r1", torch::nn::ReLU()},
            {"rnn_hxx_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)}, 
            {"rnn_hxx_n2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_hxx_l3", torch::nn::Linear(this -> _hidden, this -> _xrec)}
    }); 

    this -> rnn_top_edge = new torch::nn::Sequential({
            {"rnn_top_l1", torch::nn::Linear(this -> _xrec*4 + this -> _xout, this -> _hidden)}, 
            {"rnn_top_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_top_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)}, 
            {"rnn_top_s2", torch::nn::Sigmoid()},
            {"rnn_top_l3", torch::nn::Linear(this -> _hidden, this -> _xout)}
    }); 

    int dxx_r = this -> _xrec*4; 
    this -> rnn_res_edge = new torch::nn::Sequential({
            {"rnn_res_l1", torch::nn::Linear(dxx_r, this -> _hidden)}, 
            {"rnn_res_r1", torch::nn::ReLU()},
            {"rnn_res_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _hidden}))}, 
            {"rnn_res_l2", torch::nn::Linear(this -> _hidden, this -> _hidden)}, 
            {"rnn_res_t2", torch::nn::Sigmoid()},
            {"rnn_res_l3", torch::nn::Linear(this -> _hidden, this -> _xout)}
    }); 

    this -> mlp_ntop = new torch::nn::Sequential({
            {"ntop_l1", torch::nn::Linear(this -> _xtop + this -> _xrec, this -> _xrec)}, 
            {"ntop_r1", torch::nn::ReLU()},
            {"ntop_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xrec}))}, 
            {"ntop_l2", torch::nn::Linear(this -> _xrec, this -> _xrec)}, 
            {"ntop_t2", torch::nn::Sigmoid()},
            {"ntop_l3", torch::nn::Linear(this -> _xrec, this -> _xtop)}
    }); 

    this -> mlp_sig = new torch::nn::Sequential({
            {"res_l1", torch::nn::Linear(this -> _xout*2 + dxx_r + this -> _xtop*2, this -> _xrec*2)}, 
            {"res_r1", torch::nn::ReLU()},
            {"res_n1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({this -> _xrec*2}))}, 
            {"res_l2", torch::nn::Linear(this -> _xrec*2, this -> _xrec)}, 
            {"res_t2", torch::nn::Sigmoid()},
            {"res_l3", torch::nn::Linear(this -> _xrec, this -> _xout)}
    }); 

    this -> register_module(this -> rnn_x       , mlp_init::xavier_uniform);
    this -> register_module(this -> rnn_dx      , mlp_init::xavier_uniform);
    this -> register_module(this -> rnn_hxx     , mlp_init::xavier_uniform); 
    this -> register_module(this -> rnn_top_edge, mlp_init::xavier_uniform);
    this -> register_module(this -> rnn_res_edge, mlp_init::xavier_uniform);
    this -> register_module(this -> mlp_ntop    , mlp_init::xavier_uniform);
    this -> register_module(this -> mlp_sig     , mlp_init::xavier_uniform);
}

torch::Tensor grift::message(
        torch::Tensor _trk_i, torch::Tensor _trk_j, torch::Tensor pmc, torch::Tensor hx_i, torch::Tensor hx_j
){
    std::string key_idx = "unique"; 
    std::string key_smx = "node-sum"; 

    torch::Dict<std::string, torch::Tensor> aggr; 
    torch::Tensor trk_ij = torch::cat({_trk_i, _trk_j}, {-1}); 

    aggr = pyc::graph::unique_aggregation(trk_ij, pmc); 
    torch::Tensor pmc_ij = aggr.at(key_smx); 
    torch::Tensor m_ij   = pyc::physics::cartesian::combined::M(pmc_ij);
    torch::Tensor nds_ij = (aggr.at(key_idx) > -1).sum({-1}, true);
    torch::Tensor fx_ij  = torch::cat({m_ij, pmc_ij, nds_ij, hx_i+hx_j}, {-1}); 

    aggr = pyc::graph::unique_aggregation(_trk_i, pmc); 
    torch::Tensor pmc_i = aggr.at(key_smx); 
    torch::Tensor m_i   = pyc::physics::cartesian::combined::M(pmc_i);
    torch::Tensor nds_i = (aggr.at(key_idx) > -1).sum({-1}, true); 
    torch::Tensor fx_i  = torch::cat({m_i, pmc_i, nds_i, hx_i}, {-1}); 

    aggr = pyc::graph::unique_aggregation(_trk_j, pmc); 
    torch::Tensor pmc_j = aggr.at(key_smx);
    torch::Tensor m_j   = pyc::physics::cartesian::combined::M(pmc_j);
    torch::Tensor nds_j = (aggr.at(key_idx) > -1).sum({-1}, true); 
    torch::Tensor fx_j  = torch::cat({m_j, pmc_j, nds_j, hx_j}, {-1}); 
    return (*this -> rnn_dx) -> forward(torch::cat({fx_ij, fx_i, fx_j - fx_i}, {-1}).to(torch::kFloat32)) * hx_i.softmax(-1); 
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
        this -> x_nulls = this -> x_nulls.to(src.device()); 
        this -> dx_nulls = this -> dx_nulls.to(src.device()); 
        this -> te_nulls = this -> te_nulls.to(src.device()); 
        this -> init = true;
    }

    // ------ initialize nulls -------- //
    torch::Tensor trk = torch::zeros_like(*pt).to(torch::kInt); 
    torch::Tensor null_idx = torch::zeros_like(src); 
    torch::Tensor node_rnn = this -> x_nulls.index({trk.view({-1})}).to(torch::kFloat32); 
    torch::Tensor edge_rnn = this -> dx_nulls.index({null_idx}); 
    torch::Tensor top_edge = this -> te_nulls.index({null_idx}); 
    torch::Tensor num_node = torch::ones_like(trk); 
    torch::Tensor node_i   = num_node.cumsum({0})-1;
    torch::Tensor node_i_  = num_node.cumsum({0})-1;  

    torch::Tensor pmc = pyc::transform::combined::PxPyPzE(torch::cat({*pt, *eta, *phi, *energy}, {-1})) / 1000.0; 
    node_rnn = torch::cat({pyc::physics::cartesian::combined::M(pmc), pmc, num_node, node_rnn}, {-1}); 
    node_rnn = (*this -> rnn_x) -> forward(node_rnn.to(torch::kFloat32));
    torch::Tensor hx_i  = node_rnn.index({src});
    torch::Tensor hx_j  = node_rnn.index({dst});  

    torch::Tensor hxt_i = node_rnn.index({src}); 
    torch::Tensor hxt_j = node_rnn.index({dst}); 
    edge_rnn = (*this -> rnn_hxx) -> forward(torch::cat({edge_rnn, edge_rnn, hxt_j - hxt_i, edge_rnn, top_edge}, {-1})); 

    // ------ index the edges from 0 to N^2 -1 ------ //
    unsigned int n_nodes  = data -> num_nodes;
    torch::Tensor idx_mat = torch::zeros({n_nodes, n_nodes}, src.device()).to(torch::kLong);  
    idx_mat.index_put_({src, dst}, (null_idx+1).cumsum({-1})-1); 

    torch::Tensor norm = torch::zeros_like(idx_mat); 
    norm.index_put_({src, dst}, (null_idx+1)); 

    const std::string key_idx = "cls::1::node-indices"; 
    const std::string key_smx = "cls::1::node-sum"; 
    
    torch::Tensor pmx; 
    torch::Tensor edge_index_ = edge_index.clone();  
    torch::Dict<std::string, torch::Tensor> gr_; 
    while (edge_index_.size({1})){

        // ----- use the index matrix to map the source and destination edges to the edge index ----- //
        torch::Tensor src_ = edge_index_.index({0}); 
        torch::Tensor dst_ = edge_index_.index({1});
        torch::Tensor idx = idx_mat.index({src_, dst_}); 

        // ------------------ loop states ------------------------ //
        // ------------------ create a new message --------------------- //
        torch::Tensor dx_ij = this -> message( node_i.index({src_}), node_i.index({dst_}), pmc, hxt_i, hxt_j);   
        torch::Tensor dx_i  = this -> message(node_i_.index({src_}), node_i.index({dst_}), pmc,  hx_i, hxt_j); 
        torch::Tensor hx_ij = torch::cat({dx_ij, dx_i, hx_j - hx_i, edge_rnn, top_edge.index({idx})}, {-1});

        // ----- update the top_edge prediction weights by index ------- //
        top_edge.index_put_({idx}, (*this -> rnn_top_edge) -> forward(hx_ij)); 

        // ---- check if the new prediction is simply null ---- /
        torch::Tensor msk = std::get<1>(top_edge.index({idx}).max({-1})) < 1; // 10000 - debugging purposes.; 
        if (!msk.index({msk == false}).size({0})){break;}
        hxt_i = hxt_i.index({msk});
        hxt_j = hxt_j.index({msk}); 

        // ----------- create a new intermediate state of the nodes ----------- //
        gr_ = pyc::graph::edge_aggregation(edge_index, top_edge, pmc); 
        node_i_ = gr_.at(key_idx); 
        pmx = gr_.at(key_smx); 

        // ------ protection against depleted event graphs ---------- //
        torch::Tensor skp = (norm.sum({-1}, true) > 0).view({-1}); 
        pmx = torch::cat({pyc::physics::cartesian::combined::M(pmx), pmx, (node_i_ > -1).sum({-1}, true), node_rnn}, {-1}); 
        pmx = (*this -> rnn_x) -> forward(pmx.index({skp}).to(torch::kFloat32));
        node_rnn.index_put_({skp}, pmx);
        norm.index_put_({src_, dst_}, msk*1); 

        // ------ walk to the next node (nxt) ------- //
        hx_i = node_rnn.index({src_.index({msk})}); 
        hx_j = node_rnn.index({dst_.index({msk})}); 
        edge_rnn = (*this -> rnn_hxx) -> forward(hx_ij.index({msk}));
        edge_index_ = edge_index_.index({torch::indexing::Slice(), msk});  
    }

    // ----------- compress the top data ----------- //
    gr_ = pyc::graph::edge_aggregation(edge_index, top_edge, pmc); 
    pmx = gr_.at(key_smx); 
    trk = gr_.at(key_idx); 
    num_node = (trk > -1).sum({-1}, true);

    torch::Tensor enc_tops = torch::cat({pyc::physics::cartesian::combined::M(pmx), pmx, num_node, node_rnn}, {-1});
    torch::Tensor ntops    = (*this -> rnn_x) -> forward(enc_tops.to(torch::kFloat32)) / num_node;
    torch::Tensor tmlp     = torch::zeros({event_index.size({0}), ntops.size({1})}, ntops.device()).to(ntops.dtype()); 
    tmlp.index_add_({0}, batch_index, ntops); 
    tmlp = torch::cat({tmlp, pid}, {-1}); 
    tmlp = (*this -> mlp_ntop) -> forward(tmlp.to(torch::kFloat32));

    hxt_i = node_rnn.index({src}); 
    hxt_j = node_rnn.index({dst}); 

    trk = torch::cat({trk.index({src}), trk.index({dst})}, {-1}); 
    gr_ = pyc::graph::unique_aggregation(trk, pmc); 
    num_node = (gr_.at("unique") > -1).sum({-1}, true);
    pmx = gr_.at("node-sum");

    torch::Tensor enc_res  = torch::cat({pyc::physics::cartesian::combined::M(pmx), pmx, num_node, hxt_i}, {-1}); 
    torch::Tensor node_res = (*this -> rnn_x) -> forward(enc_res.to(torch::kFloat32)) / num_node;

    torch::Tensor fx_ij  = torch::cat({node_res, ntops.index({src}), hxt_i, hxt_j - hxt_i}, {-1});
    torch::Tensor res_edge = (*this -> rnn_res_edge) -> forward(fx_ij);

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

#include <grift.h>
#include <pyc/pyc.h>

torch::nn::Linear make_Lin(long int src, long int dst = -1){
    return torch::nn::Linear(torch::nn::LinearOptions(src, (dst < 0) ? src : dst).bias(true)); 
}

torch::nn::LayerNorm make_Nrm(long int src, long int dst = -1){
    return torch::nn::LayerNorm(torch::nn::LayerNormOptions({src})); 
}

torch::nn::PReLU make_Prm(long int n){
    return torch::nn::PReLU(torch::nn::PReLUOptions().num_parameters(n)); 
}

grift::grift(){

    // create the null buffers
    this -> dx_nulls = torch::zeros({1, this -> _xrec}).to(torch::kFloat32).detach();
    this -> te_nulls = torch::zeros({1, this -> _xout}).to(torch::kFloat32).detach(); 

    int rxn = this -> _xrec; 
    this -> rnn_x = new torch::nn::Sequential({
            {"rnn_x_l1", make_Lin(this -> _xin + rxn, rxn * 2)}, 
            {"rnn_x_t2", torch::nn::LeakyReLU()    },
            {"rnn_x_l2", make_Lin(rxn * 2, rxn * 2)},  
            {"rnn_x_n2", make_Nrm(rxn * 2)         },          
            {"rnn_x_r3", torch::nn::Tanh()         }, 
            {"rnn_x_l3", make_Lin(rxn * 2, rxn)    },  
            {"rnn_x_r4", torch::nn::LeakyReLU()    },
            {"rnn_x_l4", make_Lin(rxn)         }
    }); 

    this -> rnn_dx = new torch::nn::Sequential({
            {"rnn_dx_l0", make_Lin(this -> _xin + rxn * 2, rxn * 2)}, 
            {"rnn_dx_r2", torch::nn::LeakyReLU()    },
            {"rnn_dx_l1", make_Lin(rxn * 2, rxn * 2)}, 
            {"rnn_dx_n2", make_Nrm(rxn * 2)         },          
            {"rnn_dx_l2", make_Lin(rxn * 2, rxn)    },
            {"rnn_dx_t2", torch::nn::Tanh()         }, 
            {"rnn_dx_l3", make_Lin(rxn)             }
    }); 

    this -> rnn_hxx = new torch::nn::Sequential({
            {"rnn_hxx_l1", make_Lin(rxn * 3, rxn * 2)}, 
            {"rnn_hxx_n2", make_Prm(rxn * 2)         },          
            {"rnn_hxx_l2", make_Lin(rxn * 2,  rxn)   },
            {"rnn_hxx_t2", torch::nn::Tanh()         },
            {"rnn_hxx_l3", make_Lin(rxn)             }
    }); 

    this -> rnn_txx = new torch::nn::Sequential({
            {"rnn_txx_l0", make_Lin(rxn * 3, rxn * 3)}, 
            {"rnn_txx_n2", make_Nrm(3 * rxn)         }, 
            {"rnn_txx_l1", make_Lin(rxn * 3,   rxn)  }, 
            {"rnn_txx_p1", make_Prm(rxn)             },          
            {"rnn_txx_l2", make_Lin(rxn,    this -> _xout * 2       )},
            {"rnn_txx_t2", torch::nn::LeakyReLU()                    },
            {"rnn_txx_l3", make_Lin(this -> _xout * 2, this -> _xout)}
    }); 

    this -> rnn_rxx = new torch::nn::Sequential({
            {"rnn_rxx_l0", make_Lin(rxn * 4,  3 * rxn)}, 
            {"rnn_rxx_n1", make_Nrm(rxn * 3          )},          
            {"rnn_rxx_r2", torch::nn::LeakyReLU()     }, 
            {"rnn_rxx_l2", make_Lin(rxn * 3, this -> _xout)}, 
            {"rnn_rxx_r3", torch::nn::Softmax(-1)     },
            {"rnn_rxx_l3", make_Lin(this -> _xout, this -> _xout)}
    }); 

    int ntps = this -> _xtop + this -> _xrec; 
    this -> mlp_ntop = new torch::nn::Sequential({
            {"ntop_l1", make_Lin(ntps, rxn * 2     )}, 
            {"ntop_n1", make_Nrm(rxn * 2           )}, 
            {"ntop_r1", torch::nn::LeakyReLU()      },
            {"ntop_l2", make_Lin(rxn * 2, this -> _xtop)}, 
            {"ntop_r2", torch::nn::Softmax(-1)      },
            {"ntop_l3", make_Lin(this -> _xtop, this -> _xtop)}
    }); 

    int sigs = (this -> _xout + this -> _xtop) * 2 + this -> _xrec*4;
    this -> mlp_sig = new torch::nn::Sequential({
            {"res_l1", make_Lin(sigs, this -> _hidden)}, 
            {"res_n1", make_Nrm(this -> _hidden      )}, 
            {"res_t1", torch::nn::LeakyReLU()         },
            {"res_l2", make_Lin(this -> _hidden, rxn )}, 
            {"res_t2", torch::nn::Tanh()              },
            {"res_l3", make_Lin(rxn,     this -> _xout)}
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
    torch::Tensor mass = pyc::physics::cartesian::combined::M(pmc); 
    if (!node_rnn){return torch::cat({mass, pmc, num_node}, {-1}).to(torch::kFloat32);}
    torch::Tensor nox = torch::cat({mass, pmc, num_node, *node_rnn}, {-1});
    return (*this -> rnn_x) -> forward(nox.to(torch::kFloat32)) * 1.0 / num_node.to(torch::kFloat32);  
}

torch::Tensor grift::message(
        torch::Tensor _trk_i, torch::Tensor _trk_j, 
        torch::Tensor* hx_i,  torch::Tensor* hx_j,
        torch::Tensor pmc, torch::Tensor* dnn
){
    const std::string key_idx = "unique"; 
    const std::string key_smx = "node-sum"; 

    torch::Dict<std::string, torch::Tensor> aggr; 
    aggr = pyc::graph::unique_aggregation(_trk_i, pmc); 
    torch::Tensor fx_i = this -> node_encode(aggr.at(key_smx), aggr.at(key_idx), hx_i); 

    aggr = pyc::graph::unique_aggregation(_trk_j, pmc); 
    torch::Tensor fx_j = this -> node_encode(aggr.at(key_smx), aggr.at(key_idx), hx_j); 
    
    aggr = pyc::graph::unique_aggregation(torch::cat({_trk_i, _trk_j}, {-1}), pmc); 
    torch::Tensor fx_ij = this -> node_encode(aggr.at(key_smx), aggr.at(key_idx), dnn); 
    torch::Tensor fxK   = this -> node_encode(aggr.at(key_smx), aggr.at(key_idx), nullptr); 
    return (*this -> rnn_dx) -> forward(torch::cat({fxK, fx_ij, fx_j - fx_i}, {-1})); 
}


torch::Tensor grift::recurse(
        torch::Tensor* node_i,       torch::Tensor* idx_mat,  
        torch::Tensor* edge_index_,  torch::Tensor* edge_index, 
        torch::Tensor* edge_rnn,     torch::Tensor* node_dnn,
        torch::Tensor* top_edge,     torch::Tensor* pmc,
        torch::Tensor* node_c
){
    const std::string key_smx = "cls::1::node-sum"; 
    const std::string key_idx = "cls::1::node-indices"; 
    torch::Dict<std::string, torch::Tensor> gr_; 
    torch::Tensor _pmu = *pmc; 

    // ----- use the index matrix to map the source and destination edges to the edge index ----- //
    torch::Tensor src = edge_index_ -> index({0});
    torch::Tensor dst = edge_index_ -> index({1}); 
    torch::Tensor idx = idx_mat  -> index({src, dst}); 
    torch::Tensor idc = idx_mat  -> index({dst, src}); 

    torch::Tensor hx_i   = node_dnn -> index({src});
    torch::Tensor hx_j   = node_dnn -> index({dst});
    torch::Tensor r_dx   = edge_rnn -> index({idx}); 
    torch::Tensor top_ij = top_edge -> index({idx}); 
   
    torch::Tensor nx_i   = src.view({-1, 1}); //node_c -> index({idx}); 
    torch::Tensor nx_j   = node_c -> index({idc}); 

    // =============================================================================== //
    // ------------------ create a new message --------------------- //
    torch::Tensor hx_ij = this -> message(nx_i, nx_j, &hx_i, &hx_j, _pmu, &r_dx); 

    torch::Tensor ech_i = torch::cat({hx_ij, hx_i, hx_j - hx_i}, {-1}); 
    torch::Tensor x_ij  = (*this -> rnn_hxx) -> forward(ech_i); 

    torch::Tensor e_echo = torch::cat({hx_ij, x_ij, x_ij - r_dx}, {-1}); 
    torch::Tensor top_k = (*this -> rnn_txx) -> forward(e_echo); 
    top_edge -> index_put_({idx}, top_k); 

    gr_ = pyc::graph::edge_aggregation(edge_index -> clone(), top_edge -> clone(), _pmu); 
    torch::Tensor dh_i = this -> node_encode(gr_.at(key_smx), gr_.at(key_idx), node_dnn); 
    node_dnn -> index_put_({node_i -> view({-1})} , dh_i);  

    torch::Tensor kx = edge_index -> index({1}); 
    kx = torch::cat({gr_.at(key_idx).index({kx}), (*node_c)},{-1}); 
    gr_ = pyc::graph::unique_aggregation(kx, _pmu); 
    (*node_c) = gr_.at("unique").clone();

    kx = dh_i.index({src}); 
    dh_i = this -> node_encode(gr_.at("node-sum").index({idx}), gr_.at("unique").index({idx}), &kx); 

    e_echo = torch::cat({dh_i, x_ij, x_ij - r_dx}, {-1}); 
    top_k = (*this -> rnn_txx) -> forward(e_echo); 
    edge_rnn -> index_put_({idx}, hx_j - hx_ij.softmax(-1) * dh_i);
    top_edge -> index_put_({idx}, top_k); 

    // =============================================================================== //
    torch::Tensor msk_ij = std::get<1>(top_k.max({-1})).view({-1}); 
    return edge_index_ -> index({torch::indexing::Slice(), msk_ij < 1}); 
}

void grift::forward(graph_t* data){

    // get the particle 4-vector and convert it to cartesian
    torch::Tensor batch_index  = data -> get_batch_index(this) -> view({-1}).clone(); 
    torch::Tensor event_index  = std::get<0>(torch::_unique(batch_index)); 
    torch::Tensor* pt          = data -> get_data_node("pt",     this);
    torch::Tensor* eta         = data -> get_data_node("eta",    this);
    torch::Tensor* phi         = data -> get_data_node("phi",    this);
    torch::Tensor* energy      = data -> get_data_node("energy", this);
    torch::Tensor* is_lep      = data -> get_data_node("is_lep", this); 
    torch::Tensor pmc          = pyc::transform::separate::PxPyPzE(*pt, *eta, *phi, *energy); 

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
    torch::Tensor pid = torch::cat({*num_jets, num_bjets_, *num_leps, (*met), *met_phi}, {-1});  

    // ------ index the nodes from 0 to N-1 ----- //
    if (!this -> init){
        this -> dx_nulls = this -> dx_nulls.to(src.device()); 
        this -> te_nulls = this -> te_nulls.to(src.device()); 
        this -> init = true;
    }

    // ------ initialize nulls -------- //
    torch::Tensor null_idx = torch::zeros_like(src); 
    torch::Tensor trk      = torch::zeros_like(*pt).to(torch::kInt); 
    torch::Tensor node_rnn = torch::zeros_like(this -> dx_nulls.index({trk.view({-1})}));
    torch::Tensor edge_rnn = torch::zeros_like(this -> dx_nulls.index({null_idx})); 
    torch::Tensor top_edge = torch::zeros_like(this -> te_nulls.index({null_idx})); 
    torch::Tensor num_node = torch::ones_like(trk); 
    torch::Tensor node_i   = num_node.cumsum({0})-1;
    torch::Tensor node_s   = node_i.index({src}).clone(); 

    // ------ index the edges from 0 to N^2 -1 ------ //
    unsigned int n_nodes  = data -> num_nodes;
    torch::Tensor idx_mat = -torch::ones({n_nodes, n_nodes}, src.device()).to(torch::kLong);  
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
        long nump = p_index_.size({-1});
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
    grift* md   = new grift(); 
    md -> is_mc = this -> is_mc; 
    return md; 
}

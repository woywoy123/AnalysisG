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

    int ixn = this -> _xin + rxn; 
    this -> rnn_x = new torch::nn::Sequential({
            {"rnn_x_l1", make_Lin(ixn, rxn * 2)     },      
            {"rnn_x_r1", make_Prm(rxn * 2)          },
            {"rnn_x_l2", make_Lin(rxn * 2, rxn * 2) },            
            {"rnn_x_n1", make_Nrm(rxn * 2)          },                     
            {"rnn_x_l3", make_Lin(rxn * 2, rxn)     },                
            {"rnn_x_n3", make_Prm(rxn)              },
            {"rnn_x_l4", make_Lin(rxn)              }                          
    });
    
    int txn = this -> _xrec * 2; 
    this -> rnn_dx = new torch::nn::Sequential({
            {"rnn_dx_l1", make_Lin(txn, rxn * 2) }, 
            {"rnn_dx_r1", torch::nn::LeakyReLU() },
            {"rnn_dx_l2", make_Lin(rxn * 2)      }, 
            {"rnn_dx_n1", make_Nrm(rxn * 2)      },
            {"rnn_dx_l3", make_Lin(rxn*2, rxn)   },     
            {"rnn_dx_n2", make_Prm(rxn)          },
            {"rnn_dx_l4", make_Lin(rxn)          }               
    });

    this -> rnn_hxx = new torch::nn::Sequential({
            {"rnn_hxx_l1", make_Lin(rxn * 3, rxn * 3)  }, 
            {"rnn_hxx_t1", torch::nn::Tanh()           }, 
            {"rnn_hxx_l2", make_Lin(rxn * 3, rxn * 2)  },     
            {"rnn_hxx_n2", make_Nrm(rxn * 2)           },          
            {"rnn_hxx_l3", make_Lin(rxn * 2, rxn)      }               
    });

    this -> rnn_txx = new torch::nn::Sequential({
            {"rnn_txx_l0", make_Lin(rxn * 3, rxn * 3)},        
            {"rnn_txx_n0", make_Nrm(rxn * 3)         },
            {"rnn_txx_l1", make_Lin(rxn * 3, rxn * 2)}, 
            {"rnn_txx_n1", make_Prm(rxn * 2)         },
            {"rnn_txx_l2", make_Lin(rxn * 2, rxn)    },  
            {"rnn_txx_n2", make_Nrm(rxn)             }, 
            {"rnn_txx_p3", torch::nn::Sigmoid()      }, 
            {"rnn_txx_l3", make_Lin(rxn, this -> _xout)}
    });

    this -> rnn_rxx = new torch::nn::Sequential({
            {"rnn_rxx_l0", make_Lin(rxn * 4,  rxn * 3)}, 
            {"rnn_rxx_n1", make_Prm(rxn * 3          )},          
            {"rnn_rxx_l2", make_Lin(rxn * 3, rxn  * 2)}, 
            {"rnn_rxx_r2", torch::nn::LeakyReLU()     }, 
            {"rnn_rxx_l3", make_Lin(rxn * 2, this -> _xout)}
    }); 

    int ntps = this -> _xtop + this -> _xrec; 
    this -> mlp_ntop = new torch::nn::Sequential({
            {"ntop_l1", make_Lin(ntps, rxn * 2)    }, 
            {"ntop_n1", make_Nrm(rxn * 2)          }, 
            {"ntop_r1", torch::nn::LeakyReLU()     },
            {"ntop_l2", make_Lin(rxn * 2, rxn)     }, 
            {"ntop_r2", torch::nn::Sigmoid()       },
            {"ntop_l3", make_Lin(rxn, this -> _xtop)}
    }); 

    int sigs = (this -> _xout + this -> _xtop) * 2 + this -> _xrec*4;
    this -> mlp_sig = new torch::nn::Sequential({
            {"res_l1", make_Lin(sigs, this -> _hidden)}, 
            {"res_n1", make_Nrm(this -> _hidden      )}, 
            {"res_t1", torch::nn::LeakyReLU()         },
            {"res_l2", make_Lin(this -> _hidden, rxn )}, 
            {"res_t2", torch::nn::LeakyReLU()         },
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
        torch::Tensor  trk_i,  torch::Tensor trk_j, 
        torch::Tensor*  hx_i,  torch::Tensor* hx_j,
        torch::Tensor    pmc,  torch::Tensor* dnn
){
    const std::string key_idx = "unique"; 
    const std::string key_smx = "node-sum"; 


    torch::Dict<std::string, torch::Tensor> aggr; 
    aggr = pyc::graph::unique_aggregation(trk_i, pmc); 
    torch::Tensor fx_i = this -> node_encode(aggr.at(key_smx), aggr.at(key_idx), hx_i); 

    aggr = pyc::graph::unique_aggregation(trk_j, pmc); 
    torch::Tensor fx_j = this -> node_encode(aggr.at(key_smx), aggr.at(key_idx), hx_j); 
 
    aggr = pyc::graph::unique_aggregation(torch::cat({trk_i, trk_j}, {-1}), pmc); 
    torch::Tensor fx_ij = this -> node_encode(aggr.at(key_smx), aggr.at(key_idx), dnn); 

    return (*this -> rnn_dx) -> forward(torch::cat({fx_ij, fx_j - fx_i}, {-1})); 

}


void grift::forward(graph_t* data){
    // get the particle 4-vector and convert it to cartesian
    torch::Tensor batch_index  = data -> get_batch_index(this) -> view({-1}).clone(); 
    torch::Tensor event_index  = std::get<0>(torch::_unique(batch_index)); 

    torch::Tensor edge_index   = data -> get_edge_index(this) -> to(torch::kLong); 
    torch::Tensor src          = edge_index.index({0}).view({-1}); 
    torch::Tensor dst          = edge_index.index({1}).view({-1}); 
    torch::Tensor pmc          = this -> build_pmc(data); 
    const std::string key_idx = "cls::1::node-indices"; 
    const std::string key_smx = "cls::1::node-sum"; 

    // ------ initialize nulls -------- //
    torch::Tensor null_idx = torch::zeros_like(src); 

    torch::Tensor node_rnn = torch::zeros_like(this -> dx_nulls.index({null_idx}));
    torch::Tensor edge_rnn = torch::zeros_like(this -> dx_nulls.index({null_idx})); 
    torch::Tensor top_edge = torch::zeros_like(this -> te_nulls.index({null_idx})); 

    torch::Tensor node_i  = (torch::ones_like(batch_index).cumsum({0})-1).to(torch::kLong); 
    torch::Tensor path_s  = node_i.index({src}).view({-1, 1}).clone();
    torch::Tensor path_d  = node_i.index({src}).view({-1, 1}).clone(); 

    // ------- Build indexing mapping from i -> j, 0 to N^2 - 1 ------ //
    torch::Tensor idx_mat = this -> build_IDX(data, src, dst); 
     
    torch::Dict<std::string, torch::Tensor> gr_; 
    while (true){
        torch::Tensor sls = idx_mat.index({src, src}); 
        torch::Tensor idf = idx_mat.index({dst, src}); 
        torch::Tensor mxk = (sls > -1) * (idf > -1); 

        if (this -> break_loop(mxk)){break;}
        sls = sls.index({mxk}); 
        idf = idf.index({mxk}); 

        torch::Tensor _src = this -> expand(src, sls); 
        torch::Tensor _dst = this -> expand(dst, idf); 

        torch::Tensor path_si = path_s.index({sls}); 
        torch::Tensor path_ij = path_d.index({idf}); 

        torch::Tensor hx_i  = this -> expand(node_rnn, sls); 
        torch::Tensor hx_ij = this -> expand(edge_rnn, idf);
        torch::Tensor en_ij = this -> expand(node_rnn, idf); 

        // ------- Generate a message based on current nodes ---------- //
        torch::Tensor hk_ij = this -> message(path_si, path_ij, &hx_i, &hx_ij, pmc, &en_ij); 

        // ------- Encode the echo heard by the current  
        torch::Tensor e_di = this -> get_diff(hx_i, hk_ij, en_ij); // <- echo of j:  j -> i
        torch::Tensor e_dj = this -> get_diff(hx_i, hx_ij, en_ij); // <- echo of i:  i -> j

        torch::Tensor es_di = (*this -> rnn_hxx) -> forward(e_di); 
        torch::Tensor es_dj = (*this -> rnn_hxx) -> forward(e_dj); 

        torch::Tensor es_sx  = es_di.sigmoid() * hx_i + es_dj.sigmoid() * en_ij; 
        torch::Tensor echo_s = torch::cat({hk_ij, en_ij, es_sx}, {-1}); 

        // ------- Make a prediction -------- // 
        torch::Tensor txp    = (*this -> rnn_txx) -> forward(echo_s); 
        torch::Tensor msk_ij =   this -> get_value(txp).view({-1, 1}); 
        top_edge.index_put_({idf}, txp); 

        msk_ij  = path_s.index({sls}) * (msk_ij > 0) - 1 * (msk_ij < 1); // if true i -> j: [i + j]
        path_ij = torch::cat({path_ij, msk_ij}, {-1}).to(torch::kLong); 

        // ------ Exchange signatures -------- //
        path_d = torch::cat({path_d, path_s.index_put({idf}, msk_ij)}, {-1}).to(torch::kLong); 
        gr_ = pyc::graph::unique_aggregation(path_d, pmc); 

        // ------ encode the state of this new path (globally) ------ // 
        gr_ = pyc::graph::edge_aggregation(edge_index, top_edge, pmc); 
        torch::Tensor hxt_pi = this -> node_encode(this -> expand(gr_.at(key_smx), _src), path_si, &es_di); 
        torch::Tensor hxt_pj = this -> node_encode(this -> expand(gr_.at(key_smx), _dst), path_ij, &es_dj); 
        torch::Tensor pki = gr_.at(key_idx);

        // ----- encode the state of a single node contraction ----- //
        gr_ = pyc::graph::unique_aggregation(path_ij, pmc);
        torch::Tensor het_pi = this -> node_encode(gr_.at("node-sum"), path_si, &hxt_pi); 
        torch::Tensor het_pj = this -> node_encode(gr_.at("node-sum"), path_ij, &hxt_pj); 

        torch::Tensor hx_sg  = hxt_pi.sigmoid() * het_pi + hxt_pj.sigmoid() * het_pj; 
        torch::Tensor echo_p = torch::cat({hx_sg, hk_ij - hx_ij, es_sx}, {-1}); 
        edge_rnn.index_put_({idf}, (*this -> rnn_hxx) -> forward(echo_p)); 
        node_rnn.index_put_({idf}, hx_sg); 

        msk_ij  = this -> get_value(top_edge) > 0; 
        _src    = this -> expand(src, msk_ij); 
        _dst    = this -> expand(dst, msk_ij); 
        msk_ij  = this -> expand(msk_ij, msk_ij).to(torch::kLong); 

        torch::Tensor deP = idx_mat.index({_src, _dst});
        idx_mat.index_put_({_src, _dst}, -1 * msk_ij); 
        torch::Tensor deN = idx_mat.index({_src, _dst}); 

        torch::Tensor delta = deP != deN; 
        if (this -> break_loop(delta)){break;}
        path_d = pyc::graph::cycle_aggregation(pki, pmc).at("cycles").index({src}); 
        idx_mat = idx_mat.transpose(0, 1); 
    } 
    
    // ----------- Add additional context ----------- //
    // the event features
    torch::Tensor pid = this -> build_pid(data, event_index); 

    // ----------- compress the top data ----------- //
    gr_ = pyc::graph::edge_aggregation(edge_index, top_edge, pmc); 
    torch::Tensor rnn_DN = torch::zeros_like(this -> dx_nulls.index({torch::zeros_like(event_index)}));
    rnn_DN.index_add_({0}, batch_index.index({src}), edge_rnn * node_rnn);
    
    torch::Tensor node_trk = gr_.at(key_idx); 
    torch::Tensor ntops = this -> node_encode(gr_.at(key_smx).index({src}), node_trk.index({src}), &node_rnn); 
    torch::Tensor tmlp  = torch::zeros({event_index.size({0}), ntops.size({1})}, ntops.device()).to(ntops.dtype()); 
    tmlp.index_add_({0}, batch_index.index({src}), ntops); 

    tmlp = torch::cat({tmlp, pid}, {-1}); 
    tmlp = (*this -> mlp_ntop) -> forward(tmlp.to(torch::kFloat32));

    gr_ = pyc::graph::unique_aggregation(torch::cat({node_trk.index({src}), node_trk.index({dst})}, {-1}), pmc); 
    torch::Tensor node_res = this -> node_encode(gr_.at("node-sum"), gr_.at("unique"), &edge_rnn);

    torch::Tensor fx_ij = this -> get_diff(ntops, edge_rnn, node_rnn); 
    fx_ij = torch::cat({node_res, fx_ij}, {-1});
    torch::Tensor res_edge = (*this -> rnn_rxx) -> forward(fx_ij);

    torch::Tensor isres_ = torch::cat({res_edge,  fx_ij, top_edge - res_edge}, {-1}); 
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
}

grift::~grift(){}
model_template* grift::clone(){
    grift* md   = new grift(); 
    md -> is_mc = this -> is_mc; 
    return md; 
}

bool          grift::break_loop(torch::Tensor inpt){return !inpt.index({ inpt.view({-1}) }).size({0});}
torch::Tensor grift::expand(torch::Tensor h1, torch::Tensor msk){return h1.index({msk});}
torch::Tensor grift::get_value(torch::Tensor inpt){return std::get<1>(inpt.max({-1})).view({-1});}
torch::Tensor grift::get_diff(torch::Tensor h1, torch::Tensor h2, torch::Tensor h3){return torch::cat({h1, h2, h2 - h3}, {-1});}

// -> src | dst => 0, 1, 2, 3, 4..
torch::Tensor grift::build_IDX(graph_t* data, torch::Tensor src, torch::Tensor dst){
    long n_nodes = data -> num_nodes;
    torch::Tensor null_idx = torch::zeros_like(src); 
    torch::Tensor idx_mat = -torch::ones({n_nodes, n_nodes}, src.device()).to(torch::kLong);  
    idx_mat.index_put_({src, dst}, (null_idx+1).cumsum({-1})-1);  
    return idx_mat; 
}

torch::Tensor grift::build_pmc(graph_t* data){
    torch::Tensor* pt      = data -> get_data_node("pt",     this);
    torch::Tensor* eta     = data -> get_data_node("eta",    this);
    torch::Tensor* phi     = data -> get_data_node("phi",    this);
    torch::Tensor* energy  = data -> get_data_node("energy", this);
    torch::Tensor  pmc     = pyc::transform::separate::PxPyPzE(*pt, *eta, *phi, *energy); 
    if (this -> init){return pmc;}
    this -> dx_nulls = this -> dx_nulls.to(pt -> device()); 
    this -> te_nulls = this -> te_nulls.to(pt -> device()); 
    this -> init = true;
    return pmc; 
}

torch::Tensor grift::build_pid(graph_t* data, torch::Tensor event_idx){
    torch::Tensor* num_jets    = data -> get_data_graph("num_jets", this); 
    torch::Tensor* num_leps    = data -> get_data_graph("num_leps", this); 
    torch::Tensor* met_phi     = data -> get_data_graph("phi"     , this);
    torch::Tensor* met         = data -> get_data_graph("met"     , this); 
    torch::Tensor num_bjet     = data -> get_data_node("is_b"     , this) -> clone(); 
    torch::Tensor batch_index  = data -> get_batch_index(this) -> view({-1}).clone(); 

    torch::Tensor num_bjets_ = torch::zeros({event_idx.size({0}), 1}, num_bjet.device()).to(num_bjet.dtype()); 
    num_bjets_.index_add_({0}, batch_index, num_bjet); 
    torch::Tensor pid = torch::cat({*num_jets, num_bjets_, *num_leps, (*met), *met_phi}, {-1});  

    if (!this -> inference_mode){return pid;}
    torch::Tensor* is_lep = data -> get_data_node("is_lep", this); 

    this -> prediction_extra("is_lep"        , *is_lep); 
    this -> prediction_extra("num_leps"      , *num_leps); 
    this -> prediction_extra("num_jets"      , *num_jets); 
    this -> prediction_extra("num_bjets"     , num_bjets_); 
    if (!this -> is_mc){return pid;}

    torch::Tensor* ntops_t  = data -> get_truth_graph("ntops"  , this); 
    torch::Tensor* signa_t  = data -> get_truth_graph("signal" , this);
    torch::Tensor* r_edge_t = data -> get_truth_edge("res_edge", this); 
    torch::Tensor* t_edge_t = data -> get_truth_edge("top_edge", this); 

    this -> prediction_extra("truth_ntops"   , *ntops_t); 
    this -> prediction_extra("truth_signal"  , *signa_t); 
    this -> prediction_extra("truth_res_edge", *r_edge_t); 
    this -> prediction_extra("truth_top_edge", *t_edge_t); 
    return pid; 
}


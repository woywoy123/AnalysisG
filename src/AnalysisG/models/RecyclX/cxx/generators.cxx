#include <utils.h>
#include <recyclx.h>
#include <pyc/pyc.h>
#include <vector>

torch::Tensor utils::get_edge(recyclx* ml, graph_t* data){
    return utils::as_l(data -> get_edge_index(ml)); 
}

torch::Tensor utils::get_batch(recyclx* ml, graph_t* data){
    return utils::format(data -> get_batch_index(ml), -1); 
}

torch::Tensor utils::get_event(recyclx* ml, graph_t* data){
    return std::get<0>(torch::_unique(utils::get_batch(ml, data))); 
}

torch::Tensor utils::build_pmc(recyclx* ml, graph_t* data){
    torch::Tensor* pt      = data -> get_data_node("pt",     ml);
    torch::Tensor* eta     = data -> get_data_node("eta",    ml);
    torch::Tensor* phi     = data -> get_data_node("phi",    ml);
    torch::Tensor* energy  = data -> get_data_node("energy", ml);
    torch::Tensor  pmc     = pyc::transform::separate::PxPyPzE(*pt, *eta, *phi, *energy); 
    if (ml -> init){return pmc;}
    ml -> dx_nulls = ml -> dx_nulls.to(pt -> device()); 
    ml -> te_nulls = ml -> te_nulls.to(pt -> device()); 
    ml -> init = true;
    return pmc; 
}

torch::Tensor utils::NRecode(
        recyclx* ml, torch::Tensor pmc, torch::Tensor num_node, torch::Tensor* node_rnn
){
    torch::Tensor feats = (num_node > -1).sum({-1}, true).to(torch::kFloat32); 
    torch::Tensor mass  = pyc::physics::cartesian::combined::M(pmc); 
    torch::Tensor nox   = torch::cat({mass, pmc, feats, *node_rnn}, {-1});
    nox = (*ml -> rnn_x) -> forward(nox.to(torch::kFloat32)); 
    return nox / feats;  
}

//// -> src | dst => 0, 1, 2, 3, 4..
//torch::Tensor recyclx::build_IDX(graph_t* data, torch::Tensor src, torch::Tensor dst){
////    long n_nodes = data -> num_nodes;
////    torch::Tensor null_idx = torch::zeros_like(src); 
////    torch::Tensor idx_mat = -torch::ones({n_nodes, n_nodes}, src.device()).to(torch::kLong);  
////    idx_mat.index_put_({src, dst}, (null_idx+1).cumsum({-1})-1);  
////    return idx_mat; 
//}
//

//torch::Tensor recyclx::build_pid(graph_t* data, torch::Tensor event_idx){
////    torch::Tensor* num_jets    = data -> get_data_graph("num_jets", this); 
////    torch::Tensor* num_leps    = data -> get_data_graph("num_leps", this); 
////    torch::Tensor* met_phi     = data -> get_data_graph("phi"     , this);
////    torch::Tensor* met         = data -> get_data_graph("met"     , this); 
////    torch::Tensor num_bjet     = data -> get_data_node("is_b"     , this) -> clone(); 
////    torch::Tensor batch_index  = data -> get_batch_index(this) -> view({-1}).clone(); 
////
////    torch::Tensor num_bjets_ = torch::zeros({event_idx.size({0}), 1}, num_bjet.device()).to(num_bjet.dtype()); 
////    num_bjets_.index_add_({0}, batch_index, num_bjet); 
////    torch::Tensor pid = torch::cat({*num_jets, num_bjets_, *num_leps, (*met), *met_phi}, {-1});  
//
////    if (!this -> inference_mode){return pid;}
////    torch::Tensor* is_lep = data -> get_data_node("is_lep", this); 
////
////    this -> prediction_extra("is_lep"        , *is_lep); 
////    this -> prediction_extra("num_leps"      , *num_leps); 
////    this -> prediction_extra("num_jets"      , *num_jets); 
////    this -> prediction_extra("num_bjets"     , num_bjets_); 
////    if (!this -> is_mc){return pid;}
////
////    torch::Tensor* ntops_t  = data -> get_truth_graph("ntops"  , this); 
////    torch::Tensor* signa_t  = data -> get_truth_graph("signal" , this);
////    torch::Tensor* r_edge_t = data -> get_truth_edge("res_edge", this); 
////    torch::Tensor* t_edge_t = data -> get_truth_edge("top_edge", this); 
////
////    this -> prediction_extra("truth_ntops"   , *ntops_t); 
////    this -> prediction_extra("truth_signal"  , *signa_t); 
////    this -> prediction_extra("truth_res_edge", *r_edge_t); 
////    this -> prediction_extra("truth_top_edge", *t_edge_t); 
//    return event_idx; 
//}

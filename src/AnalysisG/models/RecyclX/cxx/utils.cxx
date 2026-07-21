#include <pyc/pyc.h>
#include <recyclx.h>
#include <utils.h>

torch::Tensor utils::NRecode(
    recyclx* ml, torch::Tensor pmc, torch::Tensor num_node, torch::Tensor* node_rnn
){
    torch::Dict<std::string, torch::Tensor> aggr; 
    aggr     = pyc::graph::unique_aggregation(num_node, pmc); 
    pmc      = aggr.at(ml -> key_smx); 
    num_node = aggr.at(ml -> key_smi); 
    num_node = utils::as_f(&num_node); 
    torch::Tensor mass  = pyc::physics::cartesian::combined::M(pmc); 
    torch::Tensor nox   = torch::cat({mass, pmc, num_node, *node_rnn}, {-1});
    return (*ml -> rnn_x) -> forward(utils::as_f(&nox)) / num_node; 
}

torch::Tensor utils::Message(
    recyclx* ml, torch::Tensor trk_i, torch::Tensor trk_j, torch::Tensor pmc, torch::Tensor* Nenc
){
    torch::Tensor trk_ij = torch::cat({utils::format(&trk_i, -1, 1), utils::format(&trk_j, -1, 1)}, {-1}); 
    torch::Dict<std::string, torch::Tensor> aggr = pyc::graph::unique_aggregation(trk_ij, pmc); 
    torch::Tensor pmc_ij = aggr.at(ml -> key_smx); 
    torch::Tensor nds_ij = aggr.at(ml -> key_smi); 
    torch::Tensor m_ij   = pyc::physics::cartesian::combined::M(pmc_ij);
    torch::Tensor fx_ij  = torch::cat({m_ij, pmc_ij, nds_ij, *Nenc}, {-1}); 
    return (*ml -> rnn_hxx) -> forward(utils::as_f(&fx_ij)); 
}

torch::Tensor utils::NDecode(
    recyclx* ml, torch::Tensor trk_i, torch::Tensor trk_j, torch::Tensor pmc, torch::Tensor* Nenc
){
    torch::Tensor trk_ij = torch::cat({ utils::format(&trk_i, -1, 1), utils::format(&trk_j, -1, 1) }, {-1}); 
    torch::Dict<std::string, torch::Tensor> aggr = pyc::graph::unique_aggregation(trk_ij, pmc); 
    torch::Tensor pmc_ij = aggr.at(ml -> key_smx); 
    torch::Tensor nds_ij = aggr.at(ml -> key_smi); 
    torch::Tensor m_ij   = pyc::physics::cartesian::combined::M(pmc_ij);
    torch::Tensor fx_ij  = torch::cat({m_ij, pmc_ij, nds_ij, *Nenc}, {-1}); 
    return pmc_ij; // + (*ml -> autodec) -> forward(utils::as_f(&fx_ij)); 
}

torch::Tensor utils::TopEdge(
    recyclx* ml, torch::Tensor trk_i, torch::Tensor trk_j, torch::Tensor pmc, torch::Tensor* Nenc
){
    // invert direction
    torch::Tensor trk_ij = torch::cat({
        utils::format(&trk_i, -1, 1), utils::format(&trk_j, -1, 1)
    }, {-1}); 

    torch::Dict<std::string, torch::Tensor> aggr; 
    aggr = pyc::graph::unique_aggregation(trk_ij, pmc); 
    torch::Tensor pmc_ij = aggr.at(ml -> key_smx); 
    torch::Tensor nds_ij = aggr.at(ml -> key_smi); 
    torch::Tensor m_ij   = pyc::physics::cartesian::combined::M(pmc_ij);
    torch::Tensor fx_ij  = torch::cat({m_ij, pmc_ij, nds_ij, *Nenc}, {-1}); 
    return (*ml -> rnn_top_edge) -> forward(utils::as_f(&fx_ij)); 
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

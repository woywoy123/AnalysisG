#include "gnn-event.h"

gnn_event::gnn_event(){
    this -> name = "gnn_event"; 
    this -> add_leaf("signal"  , "is_res_score"); 
    this -> add_leaf("ntops"   , "ntops_score"); 
    this -> add_leaf("res_edge", "res_edge_score"); 
    this -> add_leaf("top_edge", "top_edge_score"); 
    this -> add_leaf("edge_index", "edge_index"); 

    this -> add_leaf("truth_ntops"   , "truth_ntops"   ); 
    this -> add_leaf("truth_signal"  , "truth_signal"  ); 
    this -> add_leaf("truth_res_edge", "truth_res_edge"); 
    this -> add_leaf("truth_top_edge", "truth_top_edge"); 

    this -> trees = {"nominal"}; 
    this -> register_particle(&this -> m_event_particles);
    this -> register_particle(&this -> m_reco_tops);
    this -> register_particle(&this -> m_truth_tops);
}

gnn_event::~gnn_event(){
    for (size_t x(0); x < this -> reco_zprime.size(); ++x){delete this -> reco_zprime[x];}
    for (size_t x(0); x < this -> truth_zprime.size(); ++x){delete this -> truth_zprime[x];}
}

event_template* gnn_event::clone(){return (event_template*)new gnn_event();}

void gnn_event::build(element_t* el){
    el -> get("ntops"     , &this -> pred_ntops_score); 
    el -> get("signal"    , &this -> pred_signal_score); 
    el -> get("res_edge"  , &this -> pred_res_edge_score); 
    el -> get("top_edge"  , &this -> pred_top_edge_score); 
    el -> get("edge_index", &this -> m_edge_index); 

    // truth features used to create ROC curves
    el -> get("truth_ntops"   , &this -> truth_ntops); 
    el -> get("truth_signal"  , &this -> truth_signal); 
    el -> get("truth_res_edge", &this -> truth_res_edge); 
    el -> get("truth_top_edge", &this -> truth_top_edge); 
}

void gnn_event::CompileEvent(){
    std::map<int, particle_gnn*>    particle = this -> sort_by_index(&this -> m_event_particles);
    std::map<int, top_gnn*>   top_candidates = this -> sort_by_index(&this -> m_reco_tops);
    std::map<int, top_truth*>      top_truth = this -> sort_by_index(&this -> m_truth_tops);

    this -> is_signal    = this -> pred_signal_score[0] < this -> pred_signal_score[1];  
    this -> signal_score = this -> pred_signal_score[1]; 

    std::vector<bool> res_edge, truth_res_edge_; 
    for (size_t x(0); x < this -> pred_res_edge_score.size(); ++x){
        res_edge.push_back(this -> pred_res_edge_score[x][1] > 0.5); 
        truth_res_edge_.push_back(this -> truth_res_edge[x] > 0); 
    }
 
    this -> ntops_score = this -> max(&this -> pred_ntops_score); 
    for (size_t x(0); x < 5; ++x){
        if (this -> pred_ntops_score[x] != this -> ntops_score){continue;}
        this -> ntops = int(x);
        break; 
    }

    std::vector<int> src = this -> m_edge_index[0]; 
    std::vector<int> dst = this -> m_edge_index[1]; 

    std::vector<particle_template*> res_tops = {}; 
    std::vector<particle_template*> real_res_tops = {}; 
    for (int i(0); i < src.size(); ++i){
        if (!res_edge[i]){continue;}
        res_tops.push_back(top_candidates[src[i]]); 
        res_tops.push_back(top_candidates[dst[i]]); 

        if (!truth_res_edge_[i]){continue;}
        real_res_tops.push_back(top_truth[src[i]]); 
        real_res_tops.push_back(top_truth[dst[i]]); 
    }

    if (res_tops.size()){
        zprime* res = nullptr; 
        this -> sum(&res_tops, &res); 
        this -> reco_zprime.push_back(res); 
    }

    if (real_res_tops.size()){
        zprime* res = nullptr; 
        this -> sum(&real_res_tops, &res); 
        this -> truth_zprime.push_back(res); 
    }
 
    this -> vectorize(&particle      , &this -> event_particles); 
    this -> vectorize(&top_truth     , &this -> truth_tops); 
    this -> vectorize(&top_candidates, &this -> reco_tops); 
    this -> m_edge_index = {}; 
}

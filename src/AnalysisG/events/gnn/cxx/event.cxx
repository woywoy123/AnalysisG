#include "gnn-event.h"

gnn_event::gnn_event(){
    this -> name = "gnn_event"; 
    this -> add_leaf("signal"  , "is_res_score"); 
    this -> add_leaf("ntops"   , "ntops_score"); 
    this -> add_leaf("res_edge", "res_edge_score"); 
    this -> add_leaf("top_edge", "top_edge_score"); 
    this -> add_leaf("edge_index", "edge_index"); 

    this -> trees = {"nominal"}; 
    this -> register_particle(&this -> m_event_particles);
    this -> register_particle(&this -> m_reco_tops);
    this -> register_particle(&this -> m_truth_tops);
}

gnn_event::~gnn_event(){
    for (size_t x(0); x < this -> resonance.size(); ++x){
        delete this -> resonance[x];
    }
}

event_template* gnn_event::clone(){return (event_template*)new gnn_event();}

void gnn_event::build(element_t* el){
    el -> get("signal"    , &this -> m_res_score); 
    el -> get("ntops"     , &this -> m_ntops_score); 
    el -> get("res_edge"  , &this -> m_res_edge_score); 
    el -> get("top_edge"  , &this -> m_top_edge_score); 
    el -> get("edge_index", &this -> m_edge_index); 
}

void gnn_event::CompileEvent(){
    std::map<int, particle_gnn*>    particle = this -> sort_by_index(&this -> m_event_particles);
    std::map<int, top_gnn*>   top_candidates = this -> sort_by_index(&this -> m_reco_tops);
    std::map<int, top_truth*>      top_truth = this -> sort_by_index(&this -> m_truth_tops);
     
    for (size_t x(0); x < this -> m_res_edge_score.size(); ++x){
        this -> res_edge.push_back(this -> m_res_edge_score[x][1] > 0.5); 
        this -> res_score.push_back(this -> m_res_edge_score[x][1]); 
    }
    
    for (size_t x(0); x < this -> m_top_edge_score.size(); ++x){
        this -> top_edge.push_back(this -> m_top_edge_score[x][1] > 0.5); 
        this -> top_score.push_back(this -> m_top_edge_score[x][1]); 
    }

    float not_res = this -> m_res_score[0][0]; 
    float is_res  = this -> m_res_score[0][1]; 
    if (is_res > not_res){this -> is_signal = true;}
    this -> signal_score = is_res; 
  
    float max_ = this -> max(&this -> m_ntops_score[0]); 
    for (size_t x(0); x < 5; ++x){
        if (this -> m_ntops_score[0][x] != max_){continue;}
        this -> ntops = int(x);
        this -> ntops_score = max_; 
        break; 
    }

    std::vector<int> src = this -> m_edge_index[0]; 
    std::vector<int> dst = this -> m_edge_index[1]; 

    std::vector<top_gnn*> res_tops; 
    for (int i(0); i < src.size(); ++i){
        if (!this -> res_edge[i]){continue;}
        res_tops.push_back(top_candidates[src[i]]); 
        res_tops.push_back(top_candidates[dst[i]]); 
    }

    if (res_tops.size()){
        zprime* res = nullptr; 
        this -> sum(&res_tops, &res); 
        this -> resonance.push_back(res); 
    }
   
    this -> vectorize(&particle      , &this -> event_particles); 
    this -> vectorize(&top_truth     , &this -> truth_tops); 
    this -> vectorize(&top_candidates, &this -> reco_tops); 

    this -> m_edge_index = {}; 
    this -> m_res_edge_score = {}; 
    this -> m_top_edge_score = {}; 
    this -> m_ntops_score = {}; 
    this -> m_res_score = {}; 
}

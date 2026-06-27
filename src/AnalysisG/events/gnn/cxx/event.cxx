#include <reconstruction/pagerank.h>
#include <inference/gnn-event.h>

gnn_event::gnn_event(){
    this -> name = "gnn_event"; 

    // ------ observables ------- //
    this -> add_leaf("weight"    , "event_weight");
    this -> add_leaf("edge_index", "edge_index");
    this -> add_leaf("res_edge"  , "extra_res_edge_score"); 
    this -> add_leaf("top_edge"  , "extra_top_edge_score"); 
    this -> add_leaf("ntops"     , "extra_ntops_score"); 
    this -> add_leaf("signal"    , "extra_is_res_score"); 
    this -> add_leaf("num_jets"  , "extra_num_jets");  
    this -> add_leaf("num_bjets" , "extra_num_bjets");  
    this -> add_leaf("num_leps"  , "extra_num_leps"); 
    this -> add_leaf("met"       , "g_i_met");
    this -> add_leaf("phi"       , "g_i_phi");

    // ------ truth features ------ //
    this -> add_leaf("truth_ntops"   , "extra_truth_ntops"   ); 
    this -> add_leaf("truth_signal"  , "extra_truth_signal"  ); 
    this -> add_leaf("truth_res_edge", "extra_truth_res_edge"); 
    this -> add_leaf("truth_top_edge", "extra_truth_top_edge"); 

    this -> trees = {"nominal"}; 
    this -> register_particle(&this -> m_event_particles);
}

gnn_event::~gnn_event(){
    this -> deregister_particle(&this -> m_zprime[pagerank_e::truth]); 
    this -> deregister_particle(&this -> m_zprime[pagerank_e::nominal]); 
    this -> deregister_particle(&this -> m_zprime[pagerank_e::masked]); 
    this -> deregister_particle(&this -> m_zprime[pagerank_e::unmasked]); 
    this -> deregister_particle(&this -> m_zprime[pagerank_e::bias_masked]); 
    this -> deregister_particle(&this -> m_zprime[pagerank_e::bias_unmasked]); 

    this -> deregister_particle(&this -> m_tops[pagerank_e::truth]); 
    this -> deregister_particle(&this -> m_tops[pagerank_e::nominal]); 
    this -> deregister_particle(&this -> m_tops[pagerank_e::masked]); 
    this -> deregister_particle(&this -> m_tops[pagerank_e::unmasked]); 
    this -> deregister_particle(&this -> m_tops[pagerank_e::bias_masked]); 
    this -> deregister_particle(&this -> m_tops[pagerank_e::bias_unmasked]); 
}

event_template* gnn_event::clone(){return (event_template*)new gnn_event();}



void gnn_event::build(element_t* el){
    el -> get("edge_index" , &this ->    m_edge_index);
    el -> get("res_edge"   , &this -> edge_res_scores); 
    el -> get("top_edge"   , &this -> edge_top_scores); 

    reduce(el, "weight"    , &this -> weight); 
    reduce(el, "ntops"     , &this -> ntops_scores, -1);
    reduce(el, "signal"    , &this -> signal_scores, -1);
    reduce(el, "num_jets"  , &this -> num_jets);
    reduce(el, "num_bjets" , &this -> num_bjets);
    reduce(el, "num_leps"  , &this -> num_leps); 
    reduce(el, "met"       , &this -> met); 
    reduce(el, "phi"       , &this -> phi); 

    reduce(el, "truth_ntops"   , &this -> t_ntops); 
    reduce(el, "truth_signal"  , &this -> t_signal); 
    reduce(el, "truth_res_edge", &this -> t_edge_res, 0); 
    reduce(el, "truth_top_edge", &this -> t_edge_top, 0); 
}

void gnn_event::CompileEvent(){
    this -> p_signal = this -> signal_scores[0] < this -> signal_scores[1];  
    this -> s_signal = this -> signal_scores[1]; 
    this -> s_ntops  = this -> max(&this -> ntops_scores); 

    for (size_t x(0); x < 5; ++x){
        if (this -> s_ntops != this -> ntops_scores[x]){continue;}
        this -> p_ntops = int(x);
        break; 
    }

    std::map<int, std::map<int, float>> bin_top, bin_zprime, bias_zprime; 
    std::map<int, std::map<int, float>> w_nrm_top, w_nrm_zprime; 

    std::map<int, std::map<std::string, particle_gnn*>> real_tops; 
    std::map<int, std::map<std::string, particle_gnn*>> real_zprime; 

    std::map<int, std::map<std::string, particle_gnn*>> reco_tops; 
    std::map<int, std::map<std::string, particle_gnn*>> reco_zprime; 

    std::map<int, std::map<std::string, particle_gnn*>> nrm_tops;
    std::map<int, std::map<std::string, particle_gnn*>> nrm_zprime;

    std::map<int, std::map<std::string, particle_gnn*>> nom_tops; 
    std::map<int, std::map<std::string, particle_gnn*>> nom_zprime; 

    std::map<int, particle_gnn*> particle = this -> return_by_index(&this -> m_event_particles);
    for (size_t x(0); x < this -> m_edge_index.size(); ++x){
        int src = this -> m_edge_index[x][0]; 
        int dst = this -> m_edge_index[x][1];  
        particle_gnn* ptr = particle[dst]; 

        int top_ij = (this -> edge_top_scores[x][0] < this -> edge_top_scores[x][1]); 
        int res_ij = (this -> edge_res_scores[x][0] < this -> edge_res_scores[x][1]); 

        if (!reco_tops.count(src)){reco_tops[src] = {};}
        if (!reco_zprime.count(src)){reco_zprime[src] = {};}

        std::string hx       = ptr -> hash; 

        nrm_tops    [src][hx]  = ptr; 
        nrm_zprime  [src][hx]  = ptr; 
        w_nrm_zprime[src][dst] = this -> edge_res_scores[x][1];
        w_nrm_top   [src][dst] = this -> edge_top_scores[x][1];

        if (top_ij){reco_tops[src][hx] = ptr;}
        if (res_ij){reco_zprime[src][hx] = ptr;}

        bin_top    [src][dst] = this -> edge_top_scores[x][1] * top_ij;
        bin_zprime [src][dst] = this -> edge_res_scores[x][1] * top_ij;
        bias_zprime[src][dst] = this -> edge_top_scores[x][1] + this -> edge_res_scores[x][1]; 

        if (this -> t_edge_top[x]){real_tops[src][hx]   = ptr;}
        if (this -> t_edge_res[x]){real_zprime[src][hx] = ptr;}

        if (top_ij){nom_tops[src][hx] = ptr;}
        if (res_ij){nom_zprime[src][hx] = ptr;}      
    }

    // ----- vectorize the output particles ------ //
    this -> build_particles(&real_zprime, nullptr, &this -> m_zprime, false, pagerank_e::truth); 
    this -> build_particles(&real_tops  , nullptr, &this -> m_tops  , false, pagerank_e::truth); 

    // ----------- without pagerank applied ---------- //
    this -> build_particles(&nom_zprime, nullptr, &this -> m_zprime, false, pagerank_e::nominal); 
    this -> build_particles(&nom_tops  , nullptr, &this -> m_tops  , false, pagerank_e::nominal); 

    // ----------- with pagerank applied but boolean masking ---------- // 
    this -> build_particles(&reco_zprime, &bias_zprime, &this -> m_zprime, true, pagerank_e::bias_masked); 
    this -> build_particles(&reco_zprime, &bin_zprime , &this -> m_zprime, true, pagerank_e::masked); 
    this -> build_particles(&reco_tops  , &bin_top    , &this -> m_tops  , true, pagerank_e::masked); 
    
    // ----------- with pagerank applied but non boolean masking ---------- // 
    this -> build_particles(&nrm_zprime, &bias_zprime , &this -> m_zprime, true, pagerank_e::bias_unmasked); 
    this -> build_particles(&nrm_zprime, &w_nrm_zprime, &this -> m_zprime, true, pagerank_e::unmasked); 
    this -> build_particles(&nrm_tops  , &w_nrm_top   , &this -> m_tops  , true, pagerank_e::unmasked); 

    this -> event_particles = this -> vectorize(&this -> m_event_particles); 
    this -> m_edge_index = {}; 
}





#include "gnn-event.h"

gnn_event::gnn_event(){
    this -> name = "gnn_event"; 
    // ------ observables ------- //
    this -> add_leaf("signal"    , "is_res_score"); 
    this -> add_leaf("ntops"     , "ntops_score"); 
    this -> add_leaf("res_edge"  , "res_edge_score"); 
    this -> add_leaf("top_edge"  , "top_edge_score"); 
    this -> add_leaf("edge_index", "edge_index");
    this -> add_leaf("num_jets"  , "num_jets");  
    this -> add_leaf("num_bjets" , "num_bjets");  
    this -> add_leaf("num_leps"  , "num_leps");  
    this -> add_leaf("weight"    , "event_weight");

    // ------ truth features ------ //
    this -> add_leaf("truth_ntops"   , "truth_ntops"   ); 
    this -> add_leaf("truth_signal"  , "truth_signal"  ); 
    this -> add_leaf("truth_res_edge", "truth_res_edge"); 
    this -> add_leaf("truth_top_edge", "truth_top_edge"); 

    this -> trees = {"nominal"}; 
    this -> register_particle(&this -> m_event_particles);
}

gnn_event::~gnn_event(){
    this -> deregister_particle(&this -> m_r_zprime); 
    this -> deregister_particle(&this -> m_t_zprime); 
    this -> deregister_particle(&this -> m_r_tops);
    this -> deregister_particle(&this -> m_t_tops); 
}

event_template* gnn_event::clone(){return (event_template*)new gnn_event();}

void gnn_event::build(element_t* el){
    el -> get("ntops"     , &this -> ntops_scores); 
    el -> get("signal"    , &this -> signal_scores); 
    el -> get("res_edge"  , &this -> edge_res_scores); 
    el -> get("top_edge"  , &this -> edge_top_scores); 
    el -> get("edge_index", &this -> m_edge_index);

    std::vector<double> t_; 
    el -> get("num_jets", &t_); 
    this -> num_jets = t_[0]; 

    el -> get("num_leps", &t_); 
    this -> num_leps = t_[0]; 

    el -> get("weight", &t_);  
    this -> weight = t_[0]; 

    std::vector<long> x_; 
    el -> get("num_bjets", &x_);
    this -> num_bjets = x_[0]; 

    // truth features used to create ROC curves
    std::vector<int> i_;
    el -> get("truth_ntops", &i_); 
    this -> t_ntops = i_[0]; 

    std::vector<bool> b_; 
    el -> get("truth_signal", &b_); 
    this -> t_signal = b_[0]; 

    el -> get("truth_res_edge", &this -> t_edge_res); 
    el -> get("truth_top_edge", &this -> t_edge_top); 
}

void gnn_event::CompileEvent(){

    auto cluster = [this](
            std::map<int, std::map<std::string, particle_gnn*>>* clust, 
            std::map<std::string, std::vector<particle_gnn*>>* out,
            std::map<std::string, float>* bin_out,
            std::map<int, std::vector<float>>* bin_data
    ){

        std::map<std::string, std::vector<float>> score_av; 
        std::map<int, std::map<std::string, particle_gnn*>>::iterator itr = clust -> begin(); 
        for (; itr != clust -> end(); ++itr){
            if (itr -> second.size() <= 2){continue;}
            std::string hsh = ""; 
            std::map<std::string, particle_gnn*>::iterator ix = itr -> second.begin();
            for (; ix != itr -> second.end(); ++ix){ hsh = tools().hash(hsh + std::string(ix -> second -> hash)); }
            if (bin_out){
                for (float k : (*bin_data)[itr -> first]){ score_av[hsh].push_back(k); }
            }
            if (out -> count(hsh)){continue;}
            this -> vectorize(&itr -> second, &(*out)[hsh]); 
        }
        std::map<std::string, std::vector<float>>::iterator itx;
        for (itx = score_av.begin(); itx != score_av.end(); ++itx){
            float p = 0; 
            for (float k : itx -> second){p += k;}
            (*bin_out)[itx -> first] = p / float(itx -> second.size()); 
        }
    }; 


    std::map<int, particle_gnn*> particle = this -> sort_by_index(&this -> m_event_particles);

    this -> p_signal = this -> signal_scores[0] < this -> signal_scores[1];  
    this -> s_signal = this -> signal_scores[1]; 
    this -> s_ntops  = this -> max(&this -> ntops_scores); 

    for (size_t x(0); x < 5; ++x){
        if (this -> s_ntops != this -> ntops_scores[x]){continue;}
        this -> p_ntops = int(x);
        break; 
    }


    std::map<int, std::vector<float>> bin_top, bin_zprime; 
    std::map<int, std::map<std::string, particle_gnn*>> real_tops; 
    std::map<int, std::map<std::string, particle_gnn*>> real_zprime; 

    std::map<int, std::map<std::string, particle_gnn*>> reco_tops; 
    std::map<int, std::map<std::string, particle_gnn*>> reco_zprime; 

    std::vector<int> src = this -> m_edge_index[0]; 
    std::vector<int> dst = this -> m_edge_index[1]; 
    for (size_t x(0); x < src.size(); ++x){
        int top_ij = (this -> edge_top_scores[x][0] < this -> edge_top_scores[x][1]); 
        int res_ij = (this -> edge_res_scores[x][0] < this -> edge_res_scores[x][1]); 

        if (top_ij){bin_top[src[x]].push_back(this -> edge_top_scores[x][1]);}
        if (res_ij){bin_zprime[src[x]].push_back(this -> edge_res_scores[x][1]);}

        std::string hx = particle[dst[x]] -> hash; 
        if (top_ij){reco_tops[src[x]][hx]   = particle[dst[x]];}
        if (res_ij){reco_zprime[src[x]][hx] = particle[dst[x]];}
        
        if (this -> t_edge_top[x]){real_tops[src[x]][hx]   = particle[dst[x]];}
        if (this -> t_edge_res[x]){real_zprime[src[x]][hx] = particle[dst[x]];}
    }

    std::map<std::string, std::vector<particle_gnn*>>::iterator it;

    // ---- truth --- //
    std::map<std::string, std::vector<particle_gnn*>> c_real_tops;
    cluster(&real_tops  , &c_real_tops, nullptr, nullptr); 
    for (it = c_real_tops.begin(); it != c_real_tops.end(); ++it){
        top* t = nullptr;
        this -> sum(&it -> second, &t);  
        this -> m_t_tops[t -> hash] = t; 

        std::map<std::string, particle_template*> ch = t -> children; 
        std::map<std::string, particle_template*>::iterator itc = ch.begin(); 
        for (; itc != ch.end(); ++itc){t -> is_lep += ((particle_gnn*)itc -> second) -> is_lep;}
    }

    std::map<std::string, std::vector<particle_gnn*>> c_real_zprime;
    cluster(&real_zprime, &c_real_zprime, nullptr, nullptr); 
    for (it = c_real_zprime.begin(); it != c_real_zprime.end(); ++it){
        zprime* t = nullptr;
        this -> sum(&it -> second, &t);  
        this -> m_t_zprime[t -> hash] = t; 
    }

 
    // ---- reco ---- //
    std::map<std::string, float> c_reco_tops_bin; 
    std::map<std::string, std::vector<particle_gnn*>> c_reco_tops;
    cluster(&reco_tops  , &c_reco_tops, &c_reco_tops_bin, &bin_top); 
    for (it = c_reco_tops.begin(); it != c_reco_tops.end(); ++it){
        top* t = nullptr;
        this -> sum(&it -> second, &t);  
        t -> av_score = c_reco_tops_bin[it -> first]; 
        this -> m_r_tops[t -> hash] = t;

        std::map<std::string, particle_template*> ch = t -> children; 
        std::map<std::string, particle_template*>::iterator itc = ch.begin(); 
        for (; itc != ch.end(); ++itc){t -> is_lep += ((particle_gnn*)itc -> second) -> is_lep;}
    }

    std::map<std::string, float> c_reco_zprime_bin; 
    std::map<std::string, std::vector<particle_gnn*>> c_reco_zprime;
    cluster(&reco_zprime, &c_reco_zprime, &c_reco_zprime_bin, &bin_zprime);
    for (it = c_reco_zprime.begin(); it != c_reco_zprime.end(); ++it){
        zprime* t = nullptr;
        this -> sum(&it -> second, &t);  
        t -> av_score = c_reco_zprime_bin[it -> first]; 
        this -> m_r_zprime[t -> hash] = t; 
    }

    // ----- vectorize the output particles ------ //
    this -> vectorize(&this -> m_r_zprime, &this -> r_zprime); 
    this -> vectorize(&this -> m_t_zprime, &this -> t_zprime); 

    this -> vectorize(&this -> m_r_tops, &this -> r_tops); 
    this -> vectorize(&this -> m_t_tops, &this -> t_tops); 
    this -> vectorize(&this -> m_event_particles, &this -> event_particles); 

    this -> m_edge_index = {}; 
}

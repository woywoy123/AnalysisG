#include <metrics/pagerank.h>

pagerank_metric::~pagerank_metric(){}
pagerank_metric* pagerank_metric::clone(){return new pagerank_metric();}
pagerank_metric::pagerank_metric(){this -> name = "pagerank";}

void pagerank_metric::define_variables(){
//    this -> register_output("event_pagerank_training"   , "<variable>", &this -> pagerank); 
//    this -> register_output("event_pagerank_validation" , "<variable>", &this -> pagerank); 
//    this -> register_output("event_pagerank_evaluation" , "<variable>", &this -> pagerank); 
//
//    this -> register_output("global_pagerank_training"  , "<variable>", &this -> global_pagerank); 
//    this -> register_output("global_pagerank_validation", "<variable>", &this -> global_pagerank); 
//    this -> register_output("global_pagerank_evaluation", "<variable>", &this -> global_pagerank); 
}

void pagerank_metric::event(){
//    this -> write("event_pagerank_" + this -> mode, "<variable>"  , &this -> pagerank);
//    this -> write("event_pagerank_" + this -> mode, "<variable-2>", &this -> <name_2>, true);// <--- write is true
}

void pagerank_metric::batch(){}
void pagerank_metric::end(){
//    this -> write("global_pagerank_" + this -> mode, "<variable>", &this -> global_pagerank); 
//    this -> write("global_pagerank_" + this -> mode, "<variable>", &this -> global_<name_2>, true); 
}

void pagerank_metric::pagerank(
    std::map<int, std::map<std::string, std::string>>* clust, 
    std::map<int, std::map<int, float>>* bin_data
){
    float alpha = 0.85; 
    float n_nodes = clust -> size(); 
    std::map<int, std::map<int, float>> Mij; 
    std::map<int, std::map<std::string, std::string>>::iterator itr;
    
    for (itr = clust -> begin(); itr != clust -> end(); ++itr){
        if (!bin_data){break;}
        int src = itr -> first; 
        for (size_t y(0); y < n_nodes; ++y){Mij[src][y] = (src != y)*(*bin_data)[src][y];}
    }
    
    std::map<int, float> pr_;
    for (size_t y(0); y < n_nodes; ++y){
        if (!bin_data){break;}
        float sm = 0; 
        for (size_t x(0); x < n_nodes; ++x){sm += Mij[x][y];} 
        sm = ((sm) ? 1.0/sm : 0); 
        for (size_t x(0); x < n_nodes; ++x){Mij[x][y] = ((sm) ? Mij[x][y]*sm : 1.0/n_nodes)*alpha;}
        pr_[y] = (*bin_data)[y][y]/n_nodes;  
    }
    
    int timeout = 0; 
    std::map<int, float> PR = pr_; 
    while (bin_data){
        pr_.clear(); 
        float sx = 0; 
        for (size_t src(0); src < n_nodes; ++src){
            for (size_t x(0); x < n_nodes; ++x){pr_[src] += (Mij[src][x]*PR[x]);}
            pr_[src] += (1-alpha)/n_nodes; 
            sx += pr_[src]; 
        }
        itr = clust -> begin(); 
    
        float norm = 0; 
        for (; itr != clust -> end(); ++itr){
            pr_[itr -> first] = pr_[itr -> first] / sx;
            norm += std::abs(pr_[itr -> first] - PR[itr -> first]); 
            PR[itr -> first] = pr_[itr -> first]; 
        }
        timeout += 1; 
    
        if (norm > 1e-6 && timeout < 1e6){continue;}
        norm = 0; 
        for (size_t x(0); x < n_nodes; ++x){
            float sc = 0; 
            for (size_t y(0); y < n_nodes; ++y){sc += (x != y) * Mij[x][y] * (pr_[y]);}
            PR[x] = sc; norm += sc;
        }
        if (!norm){break;}
        for (size_t x(0); x < n_nodes; ++x){PR[x] = PR[x] / norm;}
        break; 
    }
}


void pagerank_metric::define_metric(metric_t* mtx){
    this -> mode = mtx -> mode(); 

    //std::map<int, particle_gnn*> particle = this -> sort_by_index(&this -> m_event_particles);
    
    //std::map<int, std::map<int, float>> bin_top, bin_zprime; 
    //std::map<int, std::map<std::string, particle_gnn*>> real_tops; 
    //std::map<int, std::map<std::string, particle_gnn*>> real_zprime; 

    //std::map<int, std::map<std::string, particle_gnn*>> reco_tops; 
    //std::map<int, std::map<std::string, particle_gnn*>> reco_zprime; 

    //for (size_t x(0); x < this -> m_edge_index.size(); ++x){
    //    int src = this -> m_edge_index[x][0]; 
    //    int dst = this -> m_edge_index[x][1];  
    //    particle_gnn* ptr = particle[dst]; 

    //    int top_ij = (this -> edge_top_scores[x][0] < this -> edge_top_scores[x][1]); 
    //    int res_ij = (this -> edge_res_scores[x][0] < this -> edge_res_scores[x][1]); 

    //    if (!reco_tops.count(src)){reco_tops[src] = {};}
    //    if (!reco_zprime.count(src)){reco_zprime[src] = {};}

    //    std::string hx       = ptr -> hash; 
    //    if (top_ij){reco_tops[src][hx]   = ptr;}
    //    if (res_ij){reco_zprime[src][hx] = ptr;}

    //    bin_top[src][dst]    = this -> edge_top_scores[x][1];
    //    bin_zprime[src][dst] = this -> edge_res_scores[x][1];

    //    if (this -> t_edge_top[x]){real_tops[src][hx]   = ptr;}
    //    if (this -> t_edge_res[x]){real_zprime[src][hx] = ptr;}
    //    //if (x == this -> m_edge_index.size()-1){continue;}
    //}

    //std::map<std::string, std::vector<particle_gnn*>>::iterator it;
    //// ---- truth --- //
    //std::map<std::string, std::vector<particle_gnn*>> c_real_tops;
    //cluster(&real_tops  , &c_real_tops, nullptr, nullptr); 
    //for (it = c_real_tops.begin(); it != c_real_tops.end(); ++it){
    //    top* t = nullptr;
    //    this -> sum(&it -> second, &t);  
    //    this -> m_t_tops[t -> hash] = t; 

    //    std::map<std::string, particle_template*> ch = t -> children; 
    //    std::map<std::string, particle_template*>::iterator itc = ch.begin(); 
    //    for (; itc != ch.end(); ++itc){t -> n_leps += ((particle_gnn*)itc -> second) -> lep;}
    //    t -> n_nodes = ch.size(); 
    //}

    //std::map<std::string, std::vector<particle_gnn*>> c_real_zprime;
    //cluster(&real_zprime, &c_real_zprime, nullptr, nullptr); 
    //for (it = c_real_zprime.begin(); it != c_real_zprime.end(); ++it){
    //    zprime* t = nullptr;
    //    this -> sum(&it -> second, &t);  
    //    this -> m_t_zprime[t -> hash] = t; 

    //    std::map<std::string, particle_template*> ch = t -> children; 
    //    std::map<std::string, particle_template*>::iterator itc = ch.begin(); 
    //    for (; itc != ch.end(); ++itc){t -> n_leps += ((particle_gnn*)itc -> second) -> lep;}
    //    t -> n_nodes = ch.size(); 
    //}

    //// ---- reco ---- //
    //std::map<std::string, float> c_reco_tops_bin; 
    //std::map<std::string, std::vector<particle_gnn*>> c_reco_tops;
    //cluster(&reco_tops  , &c_reco_tops, &c_reco_tops_bin, &bin_top); 
    //for (it = c_reco_tops.begin(); it != c_reco_tops.end(); ++it){
    //    top* t = nullptr;
    //    this -> sum(&it -> second, &t);  
    //    t -> av_score = c_reco_tops_bin[it -> first]; 
    //    this -> m_r_tops[t -> hash] = t;

    //    std::map<std::string, particle_template*> ch = t -> children; 
    //    std::map<std::string, particle_template*>::iterator itc = ch.begin(); 
    //    for (; itc != ch.end(); ++itc){t -> n_leps += ((particle_gnn*)itc -> second) -> lep;}
    //    t -> n_nodes = ch.size(); 
    //}

    //std::map<std::string, float> c_reco_zprime_bin; 
    //std::map<std::string, std::vector<particle_gnn*>> c_reco_zprime;
    //cluster(&reco_zprime, &c_reco_zprime, &c_reco_zprime_bin, &bin_zprime);
    //for (it = c_reco_zprime.begin(); it != c_reco_zprime.end(); ++it){
    //    zprime* t = nullptr;
    //    this -> sum(&it -> second, &t);  
    //    t -> av_score = c_reco_zprime_bin[it -> first]; 
    //    this -> m_r_zprime[t -> hash] = t; 

    //    std::map<std::string, particle_template*> ch = t -> children; 
    //    std::map<std::string, particle_template*>::iterator itc = ch.begin(); 
    //    for (; itc != ch.end(); ++itc){t -> n_leps += ((particle_gnn*)itc -> second) -> lep;}
    //    t -> n_nodes = ch.size(); 
    // }

    //// ----- vectorize the output particles ------ //
    //this -> vectorize(&this -> m_r_zprime, &this -> r_zprime); 
    //this -> vectorize(&this -> m_t_zprime, &this -> t_zprime); 

    //this -> vectorize(&this -> m_r_tops, &this -> r_tops); 
    //this -> vectorize(&this -> m_t_tops, &this -> t_tops); 
    //this -> vectorize(&this -> m_event_particles, &this -> event_particles); 

    //this -> m_edge_index = {}; 
}


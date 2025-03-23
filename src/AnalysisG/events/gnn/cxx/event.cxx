#include "gnn-event.h"

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
    this -> deregister_particle(&this -> m_r_zprime); 
    this -> deregister_particle(&this -> m_t_zprime); 
    this -> deregister_particle(&this -> m_r_tops);
    this -> deregister_particle(&this -> m_t_tops); 
}

event_template* gnn_event::clone(){return (event_template*)new gnn_event();}

void gnn_event::build(element_t* el){
    el -> get("edge_index" , &this -> m_edge_index);
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

    auto cluster = [this](
            std::map<int, std::map<std::string, particle_gnn*>>* clust, 
            std::map<std::string, std::vector<particle_gnn*>>* out,
            std::map<std::string, float>* bin_out,
            std::map<int, std::map<int, float>>* bin_data
    ){

        float alpha = 0.85; 
        float n_nodes = clust -> size(); 
        std::map<int, std::map<int, float>> Mij; 
        std::map<int, std::map<std::string, particle_gnn*>>::iterator itr;

        for (itr = clust -> begin(); itr != clust -> end(); ++itr){
            if (!bin_data){break;}
            int src = itr -> first; 
            for (size_t y(0); y < n_nodes; ++y){Mij[src][y] = (src != y)*(*bin_data)[src][y];}
        }
        //std::cout << " ---------- Mij ----------- "<< std::endl;
        //this -> print(&Mij); 

        std::map<int, float> pr_;
        for (size_t y(0); y < n_nodes; ++y){
            if (!bin_data){break;}
            float sm = 0; 
            for (size_t x(0); x < n_nodes; ++x){sm += Mij[x][y];} 
            sm = ((sm) ? 1.0/sm : 0); 
            for (size_t x(0); x < n_nodes; ++x){Mij[x][y] = ((sm) ? Mij[x][y]*sm : 1.0/n_nodes)*alpha;}
            pr_[y] = (*bin_data)[y][y]/n_nodes;  
        }
        //std::cout << "---------- pr_0 --------- " << std::endl;
        //this -> print(&pr_); 

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

            //std::cout << "------------ PR_" << timeout << "----------" << std::endl;
            //this -> print(&PR);

            if (norm > 1e-6 && timeout < 1e6){continue;}
            norm = 0; 
            for (size_t x(0); x < n_nodes; ++x){
                float sc = 0; 
                for (size_t y(0); y < n_nodes; ++y){
                    //if ((*bin_data)[x][y] <= 0.5){continue;}
                    sc += (x != y) * Mij[x][y] * (pr_[y]); 
                }
                PR[x] = sc;  
                norm += sc;
            }
            if (!norm){break;}
            for (size_t x(0); x < n_nodes; ++x){PR[x] = PR[x] / norm;}
            //std::cout << "------------ PR_F-------------" << std::endl;
            //this -> print(&PR); 
            break; 
        }

        tools tl = tools(); 
        std::map<std::string, particle_gnn*> tmp; 
        std::map<std::string, particle_gnn*>::iterator itp; 
        for (itr = clust -> begin(); itr != clust -> end(); ++itr){
            int src = itr -> first; 
            if (!PR[src] && bin_data){continue;}
            for (itp = itr -> second.begin(); itp != itr -> second.end(); ++itp){
                particle_gnn* ptr = itp -> second; 
                if (bin_data && (*bin_data)[src][ptr -> index] < 0.5){continue;}
                tmp[ptr -> hash] = ptr;

                std::map<std::string, particle_gnn*> mps = (*clust)[ptr -> index]; 
                std::map<std::string, particle_gnn*>::iterator itx = mps.begin(); 
                for (; itx != mps.end(); ++itx){
                    ptr = itx -> second; 
                    if (tmp.count(ptr -> hash) || clust -> count(ptr -> index)){continue;}
                    tmp[ptr -> hash] = itx -> second;
                    mps = (*clust)[ptr -> index]; 
                    itx = mps.begin(); 
                }
            }
            if (tmp.size() <= 2){continue;}
            std::string hash = ""; 
            for (itp = tmp.begin(); itp != tmp.end(); ++itp){hash = tl.hash(hash + itp -> first);}
            if (out -> count(hash)){continue;}
            this -> vectorize(&tmp, &(*out)[hash]); 
            if (!bin_out){continue;}
            for (itp = tmp.begin(); itp != tmp.end(); ++itp){(*bin_out)[hash] += PR[itp -> second -> index];}
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

    //std::cout << "*************** EVENT **************" << std::endl;
    std::map<int, std::map<int, float>> bin_top, bin_zprime; 
    std::map<int, std::map<std::string, particle_gnn*>> real_tops; 
    std::map<int, std::map<std::string, particle_gnn*>> real_zprime; 

    std::map<int, std::map<std::string, particle_gnn*>> reco_tops; 
    std::map<int, std::map<std::string, particle_gnn*>> reco_zprime; 

    //std::cout << "========== EDGE INDEX =========== " << std::endl;
    //this -> print(&this -> m_edge_index); 

    //std::cout << "========== EDGE SCORES =========== " << std::endl;
    //this -> print(&this -> edge_top_scores); 
    
    //std::cout << "========== BIN TOP =============== " << std::endl;
    //std::cout << "["; 
    for (size_t x(0); x < this -> m_edge_index.size(); ++x){
        int src = this -> m_edge_index[x][0]; 
        int dst = this -> m_edge_index[x][1];  
        particle_gnn* ptr = particle[dst]; 

        int top_ij = (this -> edge_top_scores[x][0] < this -> edge_top_scores[x][1]); 
        int res_ij = (this -> edge_res_scores[x][0] < this -> edge_res_scores[x][1]); 

        if (!reco_tops.count(src)){reco_tops[src] = {};}
        if (!reco_zprime.count(src)){reco_zprime[src] = {};}

        std::string hx       = ptr -> hash; 
        if (top_ij){reco_tops[src][hx]   = ptr;}
        if (res_ij){reco_zprime[src][hx] = ptr;}

        bin_top[src][dst]    = this -> edge_top_scores[x][1];
        bin_zprime[src][dst] = this -> edge_res_scores[x][1];

        if (this -> t_edge_top[x]){real_tops[src][hx]   = ptr;}
        if (this -> t_edge_res[x]){real_zprime[src][hx] = ptr;}
        //std::cout << top_ij; 
        //if (x == this -> m_edge_index.size()-1){continue;}
        //std::cout << ", "; 
    }
    //std::cout << "]" << std::endl;

    //std::cout << "=========== BIN TOP MATRIX ==========" << std::endl; 
    //this -> print(&bin_top); 

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
        for (; itc != ch.end(); ++itc){t -> n_leps += ((particle_gnn*)itc -> second) -> lep;}
        t -> n_nodes = ch.size(); 
    }

    std::map<std::string, std::vector<particle_gnn*>> c_real_zprime;
    cluster(&real_zprime, &c_real_zprime, nullptr, nullptr); 
    for (it = c_real_zprime.begin(); it != c_real_zprime.end(); ++it){
        zprime* t = nullptr;
        this -> sum(&it -> second, &t);  
        this -> m_t_zprime[t -> hash] = t; 

        std::map<std::string, particle_template*> ch = t -> children; 
        std::map<std::string, particle_template*>::iterator itc = ch.begin(); 
        for (; itc != ch.end(); ++itc){t -> n_leps += ((particle_gnn*)itc -> second) -> lep;}
        t -> n_nodes = ch.size(); 
    }

    //std::cout << "============= PAGE RANK ============ " << std::endl; 
    // ---- reco ---- //
    std::map<std::string, float> c_reco_tops_bin; 
    std::map<std::string, std::vector<particle_gnn*>> c_reco_tops;
    cluster(&reco_tops  , &c_reco_tops, &c_reco_tops_bin, &bin_top); 
    //std::cout << "============= END ================= " << std::endl;
    for (it = c_reco_tops.begin(); it != c_reco_tops.end(); ++it){
        top* t = nullptr;
        this -> sum(&it -> second, &t);  
        t -> av_score = c_reco_tops_bin[it -> first]; 
        this -> m_r_tops[t -> hash] = t;

        std::map<std::string, particle_template*> ch = t -> children; 
        std::map<std::string, particle_template*>::iterator itc = ch.begin(); 
        for (; itc != ch.end(); ++itc){t -> n_leps += ((particle_gnn*)itc -> second) -> lep;}
        t -> n_nodes = ch.size(); 
    }

    std::map<std::string, float> c_reco_zprime_bin; 
    std::map<std::string, std::vector<particle_gnn*>> c_reco_zprime;
    cluster(&reco_zprime, &c_reco_zprime, &c_reco_zprime_bin, &bin_zprime);
    for (it = c_reco_zprime.begin(); it != c_reco_zprime.end(); ++it){
        zprime* t = nullptr;
        this -> sum(&it -> second, &t);  
        t -> av_score = c_reco_zprime_bin[it -> first]; 
        this -> m_r_zprime[t -> hash] = t; 

        std::map<std::string, particle_template*> ch = t -> children; 
        std::map<std::string, particle_template*>::iterator itc = ch.begin(); 
        for (; itc != ch.end(); ++itc){t -> n_leps += ((particle_gnn*)itc -> second) -> lep;}
        t -> n_nodes = ch.size(); 
     }

    // ----- vectorize the output particles ------ //
    this -> vectorize(&this -> m_r_zprime, &this -> r_zprime); 
    this -> vectorize(&this -> m_t_zprime, &this -> t_zprime); 

    this -> vectorize(&this -> m_r_tops, &this -> r_tops); 
    this -> vectorize(&this -> m_t_tops, &this -> t_tops); 
    this -> vectorize(&this -> m_event_particles, &this -> event_particles); 

    this -> m_edge_index = {}; 
}

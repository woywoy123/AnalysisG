#include <inference/gnn-event.h>

void gnn_event::build_particles(
        std::map<int, std::map<std::string, particle_gnn*>>* prtl_map, 
        std::map<int, std::map<int, float>>* bin_map, 
        std::map<pagerank_e, std::vector<top*>>* out, 
        bool use_pr, pagerank_e scr
){
    std::map<std::string, top*> unique_obj = {}; 
    std::map<std::string, std::vector<particle_gnn*>> clustered;
    std::map<std::string, float> pr = this -> cluster(prtl_map , &clustered, bin_map, use_pr, scr);

    std::map<std::string, std::vector<particle_gnn*>>::iterator it;
    for (it = clustered.begin(); it != clustered.end(); ++it){
        top* tx = this -> sum<top>(&it -> second);  
        if (!tx){continue;}
        unique_obj[tx -> hash] = tx; 
        tx -> av_score = pr[it -> first]; 
        
        std::map<std::string, particle_template*> ch = tx -> children; 
        std::map<std::string, particle_template*>::iterator itc = ch.begin(); 
        for (; itc != ch.end(); ++itc){tx -> n_leps += itc -> second -> as<particle_gnn>() -> lep;}
        tx -> n_nodes = ch.size(); 
    }
    (*out)[scr] = this -> vectorize(&unique_obj); 
} 

void gnn_event::build_particles(
        std::map<int, std::map<std::string, particle_gnn*>>* prtl_map, 
        std::map<int, std::map<int, float>>* bin_map, 
        std::map<pagerank_e, std::vector<zprime*>>* out, 
        bool use_pr, pagerank_e scr
){
    std::map<std::string, zprime*> unique_obj = {}; 
    std::map<std::string, std::vector<particle_gnn*>> clustered;
    std::map<std::string, float> pr = this -> cluster(prtl_map , &clustered, bin_map, use_pr, scr);

    std::map<std::string, std::vector<particle_gnn*>>::iterator it;
    for (it = clustered.begin(); it != clustered.end(); ++it){
        zprime* tx = this -> sum<zprime>(&it -> second);  
        if (!tx){continue;}
        unique_obj[tx -> hash] = tx; 
        tx -> av_score = pr[it -> first]; 

        std::map<std::string, particle_template*> ch = tx -> children; 
        std::map<std::string, particle_template*>::iterator itc = ch.begin(); 
        for (; itc != ch.end(); ++itc){tx -> n_leps += itc -> second -> as<particle_gnn>() -> lep;}
        tx -> n_nodes = ch.size(); 
    }
    (*out)[scr] = this -> vectorize(&unique_obj); 
} 

std::map<std::string, float> gnn_event::cluster(
        std::map<int, std::map<std::string, particle_gnn*>>* clust, 
        std::map<std::string, std::vector<particle_gnn*>>* out,
        std::map<int, std::map<int, float>>* bin_data, 
        bool use_pr, pagerank_e scr
){

    float alpha = 0.85; 
    float n_nodes = 1 / float(this -> m_event_particles.size()); 
    float tel_a = (1 - alpha) * n_nodes; 
    long  l_nodes = this -> m_event_particles.size(); 
    std::map<int, std::map<int, float>> Mij; 
    std::map<int, std::map<std::string, particle_gnn*>>::iterator itr;

    for (size_t x(0); x <  l_nodes; ++x){
        if (!use_pr){break;}
        for (size_t y(0); y < l_nodes; ++y){Mij[x][y] = (x != y)*(*bin_data)[x][y];}
    }

    std::vector<float> pr_(l_nodes, 0); 
    for (size_t y(0); y < l_nodes; ++y){
        if (!use_pr){break;}
        float sm = 0; 
        for (size_t x(0); x < l_nodes; ++x){sm += Mij[x][y];} 
        sm = ((sm) ? 1.0/sm : 0); 

        for (size_t x(0); x < l_nodes; ++x){Mij[x][y] = ((sm) ? Mij[x][y] * sm : n_nodes) * alpha;}
        pr_[y] = (*bin_data)[y][y] * n_nodes;  
    }

    int timeout = 0; 
    std::vector<float> PR = pr_; 
    while (use_pr){
        pr_ = std::vector<float>(l_nodes, 0); 
        float sx = 0; 
        for (size_t x(0); x < l_nodes; ++x){
            float sc = 0; 
            for (size_t y(0); y < l_nodes; ++y){sc += (Mij[x][y] * PR[y]);}
            pr_[x] = sc + tel_a; sx += pr_[x]; 
        }

        float norm = 0; 
        for (size_t x(0); x <  l_nodes; ++x){
            pr_[x] = pr_[x] / sx;
            norm += std::abs(pr_[x] - PR[x]); 
            PR[x] = pr_[x]; 
        }

        timeout++; 
        if (norm > 1e-6 && timeout < 1e4){continue;}

        norm = 0; 
        for (size_t x(0); x < l_nodes; ++x){
            float sc = 0; 
            for (size_t y(0); y < l_nodes; ++y){sc += (x != y) * Mij[x][y] * (pr_[y]);}
            PR[x] = sc; norm += sc;
        }
        if (!norm){break;}
        for (size_t x(0); x < l_nodes; ++x){PR[x] = PR[x] / norm;}
        break; 
    }


    // ======================= Cluster particles ============================ //
    tools tl = tools(); 
    std::map<std::string, float> output; 
    std::map<std::string, particle_gnn*>::iterator itp; 
    std::map<std::string, particle_gnn*>::iterator itx;
    for (itr = clust -> begin(); itr != clust -> end(); ++itr){
        std::map<std::string, particle_gnn*> pth = itr -> second; 
        int src = itr -> first; 

        std::map<std::string, particle_gnn*> tmp; 
        for (itp = pth.begin(); itp != pth.end(); ++itp){
            particle_gnn* ptr = itp -> second; 
            int ix = ptr -> index; 
            ptr -> pr_score[scr] = PR[ix]; 
            if (use_pr && !PR[ix]){continue;}
            tmp[ptr -> hash] = ptr;

            std::map<std::string, particle_gnn*> mps = (*clust)[ix]; 
            for (itx = mps.begin(); itx != mps.end(); ++itx){
                ptr = itx -> second; 
                int index = ptr -> index; 
                std::string hash_ = ptr -> hash; 
                if (tmp.count(hash_) || clust -> count(index)){continue;}
                mps = (*clust)[index]; 
                itx = mps.begin(); 
                tmp[hash_] = ptr;

                if (itx != mps.end()){continue;}
                break;
            }
        }
        if (tmp.size() <= 2){continue;}

        std::string hash = ""; 
        for (itp = tmp.begin(); itp != tmp.end(); ++itp){hash = tl.hash(hash + itp -> first);}
        if (out -> count(hash)){continue;}
        for (itp = tmp.begin(); itp != tmp.end(); ++itp){output[hash] += PR[itp -> second -> index];}
        (*out)[hash] = this -> vectorize(&tmp); 
    }
    return output; 
}




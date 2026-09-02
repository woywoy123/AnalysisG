#include <templates/particle_template.h>
#include <metrics/pagerank.h>
#include <metrics/accuracy.h>


void accuracy_metric::pagerank(
        std::map<int, std::map<std::string, particle_template*>>* clust, 
        std::map<std::string, std::vector<particle_template*>>* out,
        std::map<std::string, float>* bin_out,
        std::map<int, std::map<int, float>>* bin_data
){
    int s = 0; 
    int e = clust -> size(); 
    float n_nodes = 1.0 / float(clust -> size()); 

    std::map<int, std::map<int, float>> Mij; 
    std::map<int, std::map<std::string, particle_template*>>::iterator itr;

    for (itr = clust -> begin(); itr != clust -> end(); ++itr){
        int src = itr -> first; 
        for (int y(s); y < e; ++y){Mij[src][y] = (src != y)*(*bin_data)[src][y];}
    }

    std::map<int, float> pr_;
    for (int y(s); y < e; ++y){
        float sm = 0; 
        for (int x(s); x < e; ++x){sm += Mij[x][y];} 
        sm = ((sm) ? 1.0/sm : 0); 
        for (int x(s); x < e; ++x){Mij[x][y] = ((sm) ? Mij[x][y]*sm : n_nodes) * this -> alpha;}
        pr_[y] = (*bin_data)[y][y] * n_nodes;  
    }

    std::map<int, float> PR = pr_; 
    for (size_t t(0); t < this -> max_itr; ++t){
        pr_.clear(); 
        float sx = 0; 
        for (int src(s); src < e; ++src){
            for (int x(s); x < e; ++x){pr_[src] += (Mij[src][x]*PR[x]);}
            pr_[src] += (1 - this -> alpha) * n_nodes; 
            sx += pr_[src]; 
        }
         
        sx = 1.0 / sx; 
        float norm = 0; 
        for (itr = clust -> begin(); itr != clust -> end(); ++itr){
            pr_[itr -> first] = pr_[itr -> first] * sx;
            norm += std::abs(pr_[itr -> first] - PR[itr -> first]); 
            PR[itr -> first] = pr_[itr -> first]; 
        }
        if (norm > this -> norm_lim){continue;}

        norm = 0; 
        for (int x(s); x < e; ++x){
            float sc = 0; 
            for (int y(s); y < e; ++y){sc += (x != y) * Mij[x][y] * (pr_[y]);}
            PR[x] = sc; norm += sc;
        }
        if (!norm){break;}
        for (int x(s); x < e; ++x){PR[x] = PR[x] / norm;}
        break; 
    }

    for (itr = clust -> begin(); itr != clust -> end(); ++itr){
        if (!PR[itr -> first]){continue;}
        std::map<std::string, particle_template*> tmp; 
        std::map<std::string, particle_template*>::iterator itp;
        for (itp = itr -> second.begin(); itp != itr -> second.end(); ++itp){
            particle_template* ptr = itp -> second; 
            if ((*bin_data)[itr -> first][ptr -> index] < 0.5){continue;}
            tmp[ptr -> hash] = ptr;

            std::map<std::string, particle_template*> mps = (*clust)[ptr -> index]; 
            std::map<std::string, particle_template*>::iterator itx = mps.begin(); 
            for (; itx != mps.end(); ++itx){
                ptr = itx -> second; 
                if (tmp.count(ptr -> hash) || clust -> count(ptr -> index)){continue;}
                tmp[ptr -> hash] = itx -> second;
                mps = (*clust)[ptr -> index]; 
                itx = mps.begin(); 
                if (mps.size()){continue;}
                break; 
            }
        }
        if (tmp.size() <= 2){continue;}
        std::string hash = ""; 
        for (itp = tmp.begin(); itp != tmp.end(); ++itp){hash = this -> tools::hash(hash + itp -> first);}
        if (out -> count(hash)){continue;}
        for (itp = tmp.begin(); itp != tmp.end(); ++itp){(*bin_out)[hash] += PR[itp -> second -> index];}
        (*out)[hash] = this -> vectorize(&tmp); 
    }
}


void accuracy_metric::pagerank(event_idx* mtx){
    std::map<int, std::map<int, float>> bin_top;  
    std::map<int, std::map<int, particle_template*>>         real_tops; 
    std::map<int, std::map<int, particle_template*>>      nominal_tops;
    std::map<int, std::map<std::string, particle_template*>> reco_tops; 
    std::map<std::string, std::map<std::string, float>>      hash_maps; 

    std::map<int, std::map<int, float>> sc_tops;  
    std::map<int, std::map<std::string, particle_template*>> umsk_tops; 

    for (size_t x(0); x < mtx -> edge_index[0].size(); ++x){
        int src = mtx -> edge_index[0][x]; 
        int dst = mtx -> edge_index[1][x];  

        float top_0 = mtx -> top_edge_score[x][0]; 
        float top_1 = mtx -> top_edge_score[x][1]; 

        particle_template* ptr = mtx -> ptx[dst]; 
        std::string hx = ptr -> hash; 

        particle_template* ptk = mtx -> ptx[src]; 
        std::string hy = ptk -> hash; 

        umsk_tops[src][hx] = ptr; 
        sc_tops[src][dst] = top_1;
        hash_maps[hx][hy] = top_1; 

        if (mtx -> top_edge_truth[x]){real_tops[src][dst] = ptr;}
        if (top_0 >= top_1){continue;}

        bin_top[src][dst] = top_1;
        reco_tops[src][hx] = ptr; 
        reco_tops[dst][hx] = ptr;
        nominal_tops[src][dst] = ptr; 
    }

    // -------------- build truth tops --------------- //
    mtx -> truth_tops = this -> build_top(&real_tops); 

    // --------------- build nominal tops ------------------ //
    mtx -> nominal_tops = this -> build_top(&nominal_tops); 
    for (size_t x(0); x < mtx -> nominal_tops.size(); ++x){
        std::map<std::string, particle_template*> ch_ = mtx -> nominal_tops[x] -> children; 
        std::vector<particle_template*> ch = this -> vectorize(&ch_); 
        float scr = 0; 
        for (size_t y(0); y < ch.size(); ++y){
            for (size_t z(y); z < ch.size(); ++z){
                std::string src = ch[y] -> hash;
                std::string dst = ch[z] -> hash; 
                scr += hash_maps[src][dst]; 
            }
        }
        mtx -> reco_scores_nom.push_back(scr); 
    }

    // --------------- build pageranked tops ------------------ //
    std::map<std::string, float> reco_tops_pagerank = {}; 
    std::map<std::string, std::vector<particle_template*>> reco_tops_pr = {}; 
    this -> pagerank(&reco_tops, &reco_tops_pr, &reco_tops_pagerank, &bin_top); 

    std::map<std::string, float>::iterator itr = reco_tops_pagerank.begin();
    for (; itr != reco_tops_pagerank.end(); ++itr){
        particle_template* px = nullptr; 
        this -> sum(&reco_tops_pr[itr -> first], &px); 
        if (!px){continue;}
        mtx -> reco_tops_pr.push_back(px); 
        mtx -> reco_scores_pr.push_back(reco_tops_pagerank[itr -> first]); 
    }

    // --------------- build unmasked pageranked tops ------------------ //
    std::map<std::string, float> reco_tops_unpagerank = {}; 
    std::map<std::string, std::vector<particle_template*>> reco_tops_upr = {}; 
    this -> pagerank(&umsk_tops, &reco_tops_upr, &reco_tops_unpagerank, &sc_tops); 

    itr = reco_tops_unpagerank.begin();
    for (; itr != reco_tops_unpagerank.end(); ++itr){
        particle_template* px = nullptr; 
        this -> sum(&reco_tops_upr[itr -> first], &px); 
        if (!px){continue;}
        mtx -> reco_tops_upr.push_back(px); 
        mtx -> reco_scores_upr.push_back(reco_tops_unpagerank[itr -> first]); 
    }
}


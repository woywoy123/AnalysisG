#include "topefficiency.h"

topefficiency::topefficiency(){this -> name = "topefficiency";}
topefficiency::~topefficiency(){}


bool topefficiency::strategy(event_template* ev){
    gnn_event* evn = (gnn_event*)ev; 

    std::string splId   = this -> get_splits(&evn -> meta_data -> sample_name, "/"); 
    std::string dataset = this -> get_splits(&evn -> meta_data -> sample_name, "/", -2); 

    this -> evnt.weight = evn -> weight; 
    this -> evnt.truth_t = evn -> m_tops[pagerank_e::truth].size(); 
    this -> evnt.norm_t  = evn -> m_tops[pagerank_e::nominal].size();  
    this -> evnt.mask_t  = evn -> m_tops[pagerank_e::masked].size(); 
    this -> evnt.nmsk_t  = evn -> m_tops[pagerank_e::unmasked].size(); 

    this -> pred_ntops_score    = {evn -> ntops_scores}; 
    this -> truth_signal        = {evn -> t_signal};   
    this -> pred_signal_score   = {evn -> signal_scores}; 



    // ------------------ get the truth kinematics of the tops ------------------ //
    this -> top_r[pagerank_e::truth].transfer(&evn -> m_tops[pagerank_e::truth]);

    //------ check if the reconstruction was ok ---------- //
    for (size_t x(0); x < evn -> m_tops[pagerank_e::truth].size(); ++x){
        top* tt = evn -> m_tops[pagerank_e::truth][x]; // <- truth tops
        this -> check_matching(&this -> top_r[pagerank_e::nominal ], tt, &evn -> m_tops[pagerank_e::nominal ]); // Nominal 
        this -> check_matching(&this -> top_r[pagerank_e::masked  ], tt, &evn -> m_tops[pagerank_e::masked  ]); // Masking 
        this -> check_matching(&this -> top_r[pagerank_e::unmasked], tt, &evn -> m_tops[pagerank_e::unmasked]); // Unmasked
    }

    this -> zprime_r[pagerank_e::truth        ].transfer(&evn -> m_zprime[pagerank_e::truth        ]); 
    this -> zprime_r[pagerank_e::nominal      ].transfer(&evn -> m_zprime[pagerank_e::nominal      ]); 
    this -> zprime_r[pagerank_e::masked       ].transfer(&evn -> m_zprime[pagerank_e::masked       ]); 
    this -> zprime_r[pagerank_e::unmasked     ].transfer(&evn -> m_zprime[pagerank_e::unmasked     ]); 
    this -> zprime_r[pagerank_e::bias_masked  ].transfer(&evn -> m_zprime[pagerank_e::bias_masked  ]); 
    this -> zprime_r[pagerank_e::bias_unmasked].transfer(&evn -> m_zprime[pagerank_e::bias_unmasked]); 

    for (size_t x(0); x < evn -> event_particles.size(); ++x){
        particle_gnn* pxt = evn -> event_particles[x]; 
        this -> m_particles.transfer(pxt); 
        this -> scores[pagerank_e::masked       ].push_back(pxt -> pr_score[pagerank_e::masked       ]); 
        this -> scores[pagerank_e::unmasked     ].push_back(pxt -> pr_score[pagerank_e::unmasked     ]); 
        this -> scores[pagerank_e::bias_masked  ].push_back(pxt -> pr_score[pagerank_e::bias_masked  ]); 
        this -> scores[pagerank_e::bias_unmasked].push_back(pxt -> pr_score[pagerank_e::bias_unmasked]); 
    }





    //// ---------------- Truth Section ---------------- //
//    this -> truth_ntops         = std::vector<int>({evn -> t_ntops}); 
    //std::map<std::string, bool> t_top_map; 
    //std::vector<top*> truth_tops = evn -> t_tops; 
    //std::string decay_region_t = this -> decaymode(truth_tops); 
    //for (size_t x(0); x < truth_tops.size(); ++x){
    //    top* top_ = truth_tops[x]; 
    //    float mass = top_ -> mass / 1000; 
    //    std::string key = this -> region(top_ -> pt / 1000, std::abs(top_ -> eta));
    //    this -> t_topmass[key][hash].push_back(mass);
    //    t_top_map[top_ -> hash] = false; 
    //    this -> t_nodes[hash][mass] = top_ -> n_nodes; 
    //    this -> t_decay_region[decay_region_t][hash].push_back(mass); 
    //}
    //this -> n_tru_tops[hash] = t_top_map.size(); 

    //std::vector<zprime*> truth_zprime = evn -> t_zprime;  
    //for (size_t x(0); x < truth_zprime.size(); ++x){
    //    zprime* zp_ = truth_zprime[x];
    //    std::string key = this -> region(zp_ -> pt / 1000, std::abs(zp_ -> eta));
    //    this -> t_zmass[key][hash].push_back(zp_ -> mass / 1000); 
    //}

    //// ----------------- Reconstructed Section ------------------- //
    //std::vector<zprime*> reco_zprime = evn -> r_zprime;  
    //for (size_t x(0); x < reco_zprime.size(); ++x){
    //    zprime* zp_ = reco_zprime[x]; 
    //    std::string key = this -> region(zp_ -> pt / 1000, std::abs(zp_ -> eta));
    //    this -> p_zmass[key][hash].push_back(zp_ -> mass / 1000); 
    //    this -> prob_zprime[key][hash].push_back(zp_ -> av_score); 
    //}

    //std::vector<top*> reco_tops = evn -> r_tops; 
    //std::string decay_region_p = this -> decaymode(reco_tops); 
    //for (size_t x(0); x < reco_tops.size(); ++x){
    //    top* top_ = reco_tops[x]; 
    //    float mass = top_ -> mass / 1000; 
    //    std::string key = this -> region(top_ -> pt / 1000, std::abs(top_ -> eta));
    //    this -> p_topmass[key][hash].push_back(mass);
    //    this -> prob_tops[key][hash].push_back(top_ -> av_score); 
    //    this -> p_nodes[hash][mass] = top_ -> n_nodes; 
    //    this -> p_decay_region[decay_region_p][hash].push_back(mass); 
    //}

    //// ------------------ Efficiency Reconstruction ------------------- //
    //for (size_t s(0); s <= int(1.0/this -> score_step); ++s){
    //    int perf = 0; 
    //    int reco = 0; 
    //    float sc = s*this -> score_step; 
    //    for (size_t x(0); x < reco_tops.size(); ++x){
    //        top* top_ = reco_tops[x]; 
    //        std::string id = top_ -> hash; 
    //        if (top_ -> av_score <= sc){continue;}
    //        ++reco; 

    //        if (!t_top_map.count(id)){continue;}
    //        if (t_top_map[id]){continue;} // prevent double counting
    //        t_top_map[id] = true; 
    //        ++perf;
    //    }
    //    this -> n_perfect_tops[hash][sc] = perf; 
    //    this -> n_pred_tops[hash][sc]    = reco; 
    //    std::map<std::string, bool>::iterator ib = t_top_map.begin(); 
    //    for (; ib != t_top_map.end(); ++ib){ib -> second = false;}
    //}

    //this -> truth_top_edge      = evn -> t_edge_top; 
    //this -> pred_top_edge_score = {evn -> edge_top_scores}; 

    //this -> truth_res_edge      = evn -> t_edge_res; 
    //this -> pred_res_edge_score = {evn -> edge_res_scores}; 

    return true; 
}



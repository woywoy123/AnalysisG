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
    return true; 
}



#include "topefficiency.h"


void dump_t::transfer(top* ptr){
    this -> pt.push_back(ptr   -> pt * 0.001); 
    this -> mass.push_back(ptr -> mass * 0.001); 
    this -> eta.push_back(ptr  -> eta); 
    this -> phi.push_back(ptr  -> phi); 
    this -> leptonic.push_back(!ptr -> n_leps); 
}

void dump_t::transfer(particle_gnn* ptr){
    this -> pt.push_back(ptr       -> pt   * 0.001); 
    this -> mass.push_back(ptr     -> mass * 0.001); 
    this -> eta.push_back(ptr      -> eta); 
    this -> phi.push_back(ptr      -> phi); 
    this -> leptonic.push_back(ptr -> lep);
}



void dump_t::transfer(std::vector<top*>* ptr){
    for (size_t x(0); x < ptr -> size(); ++x){
        this -> transfer(ptr -> at(x));
    }
}


void dump_t::transfer(std::vector<zprime*>* ptr){
    for (size_t x(0); x < ptr -> size(); ++x){
        this -> pt.push_back(ptr   -> at(x) -> pt * 0.001); 
        this -> mass.push_back(ptr -> at(x) -> mass * 0.001); 
        this -> eta.push_back(ptr  -> at(x) -> eta); 
        this -> phi.push_back(ptr  -> at(x) -> phi); 
        this -> leptonic.push_back(ptr -> at(x) -> n_leps); 
    }
}

std::tuple<dump_t*, std::string> topefficiency::switching(
        std::map<pagerank_e, dump_t>* prx, pagerank_e px, std::string pname
){
    std::string base = ""; 
    switch (px){
        case pagerank_e::truth:         base = pname + "_truth";            break;
        case pagerank_e::nominal:       base = pname + "_nominal";          break;
        case pagerank_e::masked:        base = pname + "_PR_masked";        break;
        case pagerank_e::unmasked:      base = pname + "_PR_unmasked";      break;
        case pagerank_e::bias_masked:   base = pname + "_PR_bias_masked";   break;
        case pagerank_e::bias_unmasked: base = pname + "_PR_bias_unmasked"; break;
        default: break; 
    }
    return {&(*prx)[px], base}; 
}; 

void topefficiency::writing(
    std::map<pagerank_e, dump_t>* prx, pagerank_e px, std::string pname
){
    std::tuple<dump_t*, std::string> vs = this -> switching(prx, px, pname); 
    dump_t*      d = std::get<0>(vs); 
    std::string nm = std::get<1>(vs); 
    this -> write(&d -> pt,       nm + "_pt"      ); 
    this -> write(&d -> eta,      nm + "_eta"     ); 
    this -> write(&d -> phi,      nm + "_phi"     ); 
    this -> write(&d -> mass,     nm + "_mass"    ); 
    this -> write(&d -> leptonic, nm + "_leptonic"); 
    if (px == pagerank_e::truth){return;}
    if (pname == "zprime"){
        if (px == pagerank_e::nominal){return;}
        this -> write(&d -> ranks,    nm + "_ranks"   ); 
        return; 
    }

    this -> write(&d -> lowC,     nm + "_chi2"    ); 
    if (px == pagerank_e::nominal){return;}
    this -> write(&d -> ranks,    nm + "_ranks"   ); 
}

void topefficiency::merge(selection_template* sl){
    topefficiency* slt = (topefficiency*)sl; 

    this -> writing(&slt -> top_r   , pagerank_e::truth        , "top"); 
    this -> writing(&slt -> top_r   , pagerank_e::nominal      , "top"); 
    this -> writing(&slt -> top_r   , pagerank_e::masked       , "top"); 
    this -> writing(&slt -> top_r   , pagerank_e::unmasked     , "top"); 

    this -> writing(&slt -> zprime_r, pagerank_e::truth        , "zprime"); 
    this -> writing(&slt -> zprime_r, pagerank_e::nominal      , "zprime"); 
    this -> writing(&slt -> zprime_r, pagerank_e::masked       , "zprime"); 
    this -> writing(&slt -> zprime_r, pagerank_e::unmasked     , "zprime"); 
    this -> writing(&slt -> zprime_r, pagerank_e::bias_masked  , "zprime"); 
    this -> writing(&slt -> zprime_r, pagerank_e::bias_unmasked, "zprime"); 

    this -> write(&slt -> m_particles.pt,       "particle_pt"      ); 
    this -> write(&slt -> m_particles.eta,      "particle_eta"     ); 
    this -> write(&slt -> m_particles.phi,      "particle_phi"     ); 
    this -> write(&slt -> m_particles.mass,     "particle_mass"    ); 
    this -> write(&slt -> m_particles.leptonic, "particle_leptonic");

    this -> write(&slt -> scores[pagerank_e::masked       ], "PR_masked"); 
    this -> write(&slt -> scores[pagerank_e::unmasked     ], "PR_unmasked"); 
    this -> write(&slt -> scores[pagerank_e::bias_masked  ], "PR_bias_masked"); 
    this -> write(&slt -> scores[pagerank_e::bias_unmasked], "PR_bias_unmasked"); 

    this -> write(&slt -> evnt.truth_t     , "truth_ntops"   ); 
    this -> write(&slt -> evnt.norm_t      , "nominal_ntops" ); 
    this -> write(&slt -> evnt.mask_t      , "masked_ntops"  ); 
    this -> write(&slt -> evnt.nmsk_t      , "unmasked_ntops"); 
    this -> write(&slt -> evnt.weight      , "event_weight"); 

    this -> write(&slt -> truth_signal     , "truth_signal");   
    this -> write(&slt -> pred_ntops_score , "prediction_ntops"); 
    this -> write(&slt -> pred_signal_score, "prediction_signal");               

}


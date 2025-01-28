#include "parton.h"

parton::parton(){this -> name = "parton";}
parton::~parton(){}

selection_template* parton::clone(){
    return (selection_template*)new parton();
}

void parton::merge(selection_template* sl){
    parton* slt = (parton*)sl; 

    merge_data(&this -> ntops_tjets_pt, &slt -> ntops_tjets_pt); 
    merge_data(&this -> ntops_tjets_e , &slt -> ntops_tjets_e); 

    merge_data(&this -> ntops_jets_pt , &slt -> ntops_jets_pt); 
    merge_data(&this -> ntops_jets_e  , &slt -> ntops_jets_e); 

    merge_data(&this -> nparton_tjet_e             , &slt -> nparton_tjet_e); 
    merge_data(&this -> nparton_jet_e              , &slt -> nparton_jet_e);                   
    merge_data(&this -> frac_parton_tjet_e         , &slt -> frac_parton_tjet_e);           
    merge_data(&this -> frac_parton_jet_e          , &slt -> frac_parton_jet_e);            
    merge_data(&this -> frac_ntop_tjet_contribution, &slt -> frac_ntop_tjet_contribution);   
    merge_data(&this -> frac_ntop_jet_contribution , &slt -> frac_ntop_jet_contribution);   
    merge_data(&this -> frac_mass_top              , &slt -> frac_mass_top);                  
}

bool parton::selection(event_template* ev){
    bsm_4tops* bsm = (bsm_4tops*)ev; 
    return bsm -> Tops.size() >= 2 && bsm -> Tops.size() < 5; 
}

bool parton::strategy(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 

    std::vector<truthjet*> tjet_all; 
    std::vector<jet*> jet_all; 
    std::vector<top*> tops; 

    this -> upcast(&evn -> TruthJets, &tjet_all); 
    this -> upcast(&evn -> Jets, &jet_all); 
    this -> upcast(&evn -> Tops, &tops);

    // ------ Count the number of tops contributing to jets ------- //
    for (size_t x(0); x < tjet_all.size(); ++x){
        int ntops = this -> make_unique(&tjet_all[x] -> Tops).size(); 
        std::string key = this -> to_string(ntops) + "::tops"; 
        this -> ntops_tjets_pt[key].push_back(tjet_all[x] -> pt / 1000); 
        this -> ntops_tjets_e[key].push_back(tjet_all[x] -> e / 1000); 

        std::vector<truthjetparton*> parton_tj = tjet_all[x] -> Parton; 
        key = this -> to_string(parton_tj.size()) + "::partons";
        particle_template* tjx = nullptr;
        this -> sum(&parton_tj, &tjx); 
        this -> nparton_tjet_e[key].push_back(tjx -> e / tjet_all[x] -> e); 

        for (size_t p(0); p < parton_tj.size(); ++p){
            truthjetparton* tp = parton_tj[p]; 
            key = this -> to_string(std::abs(tp -> pdgid)); 
            this -> frac_parton_tjet_e[key].push_back(tp -> e / tjx -> e); 
        }
    }

    for (size_t x(0); x < jet_all.size(); ++x){
        int ntops = this -> make_unique(&jet_all[x] -> Tops).size(); 
        std::string key = this -> to_string(ntops) + "::tops"; 
        this -> ntops_jets_pt[key].push_back(jet_all[x] -> pt / 1000); 
        this -> ntops_jets_e[key].push_back(jet_all[x] -> e / 1000); 

        std::vector<jetparton*> parton_j = jet_all[x] -> Parton; 
        key = this -> to_string(parton_j.size()) + "::partons";
        particle_template* jx = nullptr;
        this -> sum(&parton_j, &jx); 
        this -> nparton_jet_e[key].push_back(jx -> e / jet_all[x] -> e); 

        for (size_t p(0); p < parton_j.size(); ++p){
            jetparton* tp = parton_j[p]; 
            key = this -> to_string(std::abs(tp -> pdgid)); 
            this -> frac_parton_jet_e[key].push_back(tp -> e / jx -> e); 
        }
    }

    for (size_t t(0); t < tops.size(); ++t){
        top* ti = tops[t];

        std::vector<top_children*> ch;
        this -> upcast(&ti -> children, &ch); 

        bool is_lepton = false; 
        for (size_t x(0); x < ch.size(); ++x){
            if (!ch[x] -> is_lep){continue;}
            is_lepton = true; 
            break; 
        }
        if (is_lepton){continue;}

        std::vector<truthjet*> tjets_ = ti -> TruthJets; 
        truthjetparton* p = nullptr; 

        std::vector<jet*> jets_ = ti -> Jets; 
        jetparton* p_ = nullptr; 

        this -> frac_mass_top["0.00::truthjet"].push_back(this -> top_mass_contribution(tjets_, &this -> frac_ntop_tjet_contribution, p, ti, 0));
        this -> frac_mass_top["0.00::jet"].push_back(this -> top_mass_contribution(jets_, &this -> frac_ntop_jet_contribution, p_, ti, 0));

        for (size_t x(1); x <= 100; ++x){
            float v = 0.1 * x; 
            std::string kx = this -> to_string(v, 2); 
            this -> frac_mass_top[kx + "::truthjet"].push_back(this -> top_mass_contribution(tjets_, nullptr, p, ti, v));
            this -> frac_mass_top[kx + "::jet"].push_back(this -> top_mass_contribution(jets_, nullptr, p_, ti, v));
        }
    }

    return true; 
}


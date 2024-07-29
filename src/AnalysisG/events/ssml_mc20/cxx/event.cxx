#include "event.h"

ssml_mc20::ssml_mc20(){
    this -> name = "ssml_mc20"; 
    this -> add_leaf("weight", "weight_mc_NOSYS"); 
    this -> add_leaf("met", "met_met_NOSYS"); 
    this -> add_leaf("phi", "met_phi_NOSYS"); 
    this -> add_leaf("met_sum", "met_sumet_NOSYS");
    this -> add_leaf("eventNumber", "eventNumber"); 
    this -> add_leaf("nbj_65", "nBjets_GN2v01_65WP"); 
    this -> add_leaf("nbj_70", "nBjets_GN2v01_70WP"); 
    this -> add_leaf("nbj_77", "nBjets_GN2v01_77WP"); 
    this -> add_leaf("nbj_85", "nBjets_GN2v01_85WP"); 
    this -> add_leaf("nbj_90", "nBjets_GN2v01_90WP"); 
    this -> add_leaf("nEl", "nElectrons"); 
    this -> add_leaf("nJets", "nJets"); 
    this -> add_leaf("nLeptons", "nLeptons"); 
    this -> add_leaf("nMuons", "nMuons"); 
    this -> trees = {"reco"}; 

    this -> register_particle(&this -> m_electrons); 
    this -> register_particle(&this -> m_muons); 
    this -> register_particle(&this -> m_jets); 
    this -> register_particle(&this -> m_tops);
    this -> register_particle(&this -> m_zprime); 
    this -> register_particle(&this -> m_parton1); 
    this -> register_particle(&this -> m_parton2); 
    this -> register_particle(&this -> m_bpartons); 
    this -> register_particle(&this -> m_truthjets); 

}

ssml_mc20::~ssml_mc20(){}

event_template* ssml_mc20::clone(){
    return (event_template*)new ssml_mc20();
}

void ssml_mc20::build(element_t* el){
    el -> get("weight"     , (float*)&this -> weight); 
    el -> get("met"        , &this -> met); 
    el -> get("phi"        , &this -> phi); 
    el -> get("met_sum"    , &this -> met_sum); 
    el -> get("eventNumber", &this -> eventNumber); 

    el -> get("nbj_65", &this -> n_bj_65); 
    el -> get("nbj_70", &this -> n_bj_70); 
    el -> get("nbj_77", &this -> n_bj_77); 
    el -> get("nbj_85", &this -> n_bj_85); 
    el -> get("nbj_90", &this -> n_bj_90); 

    el -> get("nEl"     , &this -> n_els); 
    el -> get("nJets"   , &this -> n_jets); 
    el -> get("nLeptons", &this -> n_leps); 
    el -> get("nMuons"  , &this -> n_mus); 
}

void ssml_mc20::CompileEvent(){
    this -> vectorize(&this -> m_tops, &this -> Tops); 
    for (size_t x(0); x < this -> Tops.size(); ++x){this -> Tops[x] -> type = "top";}
    std::map<int, top*> tops_ = this -> sort_by_index(&this -> m_tops); 

    std::vector<particle_template*> partons; 
    this -> vectorize(&this -> m_parton1, &partons); 
    this -> vectorize(&this -> m_parton2, &partons); 
    this -> vectorize(&this -> m_bpartons, &partons); 
    for (size_t x(0); x < partons.size(); ++x){
        particle_template* prt = partons[x]; 
        prt -> type = "parton"; 

        this -> TruthChildren.push_back(prt); 
        tops_[prt -> index] -> Children.push_back(prt); 
        prt -> register_parent(tops_[prt -> index]); 
        if (!prt -> is_nu){continue;}
        tops_[prt -> index] -> Neutrinos.push_back(prt);
    }
    this -> vectorize(&this -> m_jets, &this -> Jets); 
    
    std::map<std::string, jet*>::iterator itj = this -> m_jets.begin();
    for (; itj != this -> m_jets.end(); ++itj){
        jet* prt = itj -> second; 
        int idx = int(prt -> top_index); 
        if (idx < 0){continue;}
        tops_[idx] -> Jets.push_back(prt); 
        prt -> register_parent(tops_[idx]); 
        prt -> pdgid = (prt -> btag_90) ? 5 : 0; 
    } 

    std::map<std::string, electron*>::iterator ite = this -> m_electrons.begin();
    for (; ite != this -> m_electrons.end(); ++ite){
        int idx = ite -> second -> top_index; 
        this -> Leptons.push_back(ite -> second); 
        if (idx < 0){continue;}
        tops_[idx] -> Leptons.push_back(ite -> second); 
        ite -> second -> register_parent(tops_[idx]); 
    } 

    std::map<std::string, muon*>::iterator itm = this -> m_muons.begin();
    for (; itm != this -> m_muons.end(); ++itm){
        int idx = itm -> second -> top_index; 
        this -> Leptons.push_back(itm -> second); 
        if (idx < 0){continue;}
        tops_[idx] -> Leptons.push_back(itm -> second); 
        itm -> second -> register_parent(tops_[idx]); 
    } 

    this -> vectorize(&this -> m_jets     , &this -> Detector); 
    this -> vectorize(&this -> m_muons    , &this -> Detector); 
    this -> vectorize(&this -> m_electrons, &this -> Detector); 
}

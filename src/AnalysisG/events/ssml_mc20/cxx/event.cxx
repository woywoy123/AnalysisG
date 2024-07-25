#include "event.h"

ssml_mc20::ssml_mc20(){
    this -> name = "ssml_mc20"; 
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

    this -> register_particle(&this -> m_jets);
    this -> register_particle(&this -> m_leptons);
    this -> register_particle(&this -> m_electrons);
    this -> register_particle(&this -> m_muons);
}

ssml_mc20::~ssml_mc20(){}

event_template* ssml_mc20::clone(){return (event_template*)new ssml_mc20();}

void ssml_mc20::build(element_t* el){
    el -> get("met", &this -> met); 
    el -> get("phi", &this -> phi); 
    el -> get("met_sum", &this -> met_sum); 
    el -> get("eventNumber", &this -> eventNumber); 

    el -> get("nbj_65", &this -> n_bj_65); 
    el -> get("nbj_70", &this -> n_bj_70); 
    el -> get("nbj_77", &this -> n_bj_77); 
    el -> get("nbj_85", &this -> n_bj_85); 
    el -> get("nbj_90", &this -> n_bj_90); 

    el -> get("nEl", &this -> n_els); 
    el -> get("nJets", &this -> n_jets); 
    el -> get("nLeptons", &this -> n_leps); 
    el -> get("nMuons", &this -> n_mus); 

}

void ssml_mc20::CompileEvent(){
    this -> vectorize(&this -> m_jets     , &this -> Jets);  
    this -> vectorize(&this -> m_leptons  , &this -> Leptons);  
    this -> vectorize(&this -> m_electrons, &this -> Electrons);  
    this -> vectorize(&this -> m_muons    , &this -> Muons);  
}

#include "event.h"

ssml_mc20::ssml_mc20(){
    this -> name = "ssml_mc20"; 
    this -> trees = {"reco"}; 

    this -> add_leaf("eventNumber"   , "eventNumber"); 
    this -> add_leaf("event_category", "event_BkgCategory"); 
    this -> add_leaf("weight"        , "weight_mc_NOSYS"); 
    this -> add_leaf("weight_pileup" , "weight_pileup_NOSYS"); 

    this -> add_leaf("phi"    , "met_phi_NOSYS"); 
    this -> add_leaf("met"    , "met_met_NOSYS"); 
    this -> add_leaf("met_sum", "met_sumet_NOSYS");

    this -> add_leaf("nElectrons", "nElectrons"); 
    this -> add_leaf("nFJets"    , "nFJets");
    this -> add_leaf("nJets"     , "nJets"); 
    this -> add_leaf("nLeptons"  , "nLeptons"); 
    this -> add_leaf("nMuons"    , "nMuons"); 

    this -> register_particle(&this -> m_zprime); 
    this -> register_particle(&this -> m_tops);
    this -> register_particle(&this -> m_truthjets); 
    this -> register_particle(&this -> m_partons); 

    this -> register_particle(&this -> m_electrons); 
    this -> register_particle(&this -> m_muons); 
    this -> register_particle(&this -> m_jets); 
}

ssml_mc20::~ssml_mc20(){}
event_template* ssml_mc20::clone(){return (event_template*)new ssml_mc20();}

void ssml_mc20::build(element_t* el){
    el -> get("eventNumber"   , &this -> eventNumber); 
    el -> get("event_category", &this -> event_category); 
    el -> get("weight"        , (float*)&this -> weight); 
    el -> get("weight_pileup" , &this -> weight_pileup); 

    el -> get("nElectrons", &this -> n_electrons); 
    el -> get("nFJets"    , &this -> n_fjets); 
    el -> get("nJets"     , &this -> n_jets); 
    el -> get("nLeptons"  , &this -> n_leptons); 
    el -> get("nMuons"    , &this -> n_muons); 

    el -> get("met"        , &this -> met); 
    el -> get("phi"        , &this -> phi); 
    el -> get("met_sum"    , &this -> met_sum); 
}

void ssml_mc20::CompileEvent(){
    std::map<int, top*> tops_      = this -> sort_by_index(&this -> m_tops); 
    std::map<int, jet*> jets       = this -> sort_by_index(&this -> m_jets);
    std::map<int, muon*> mux       = this -> sort_by_index(&this -> m_muons); 
    std::map<int, electron*> elx   = this -> sort_by_index(&this -> m_electrons); 
    std::map<int, truthjet*> tjets = this -> sort_by_index(&this -> m_truthjets);

    this -> vectorize(&this -> m_tops  , &this -> Tops); 
    this -> vectorize(&this -> m_zprime, &this -> Zprime); 
    for (size_t x(0); x < this -> Tops.size(); ++x){
        top* t = (top*)this -> Tops[x]; 
        t -> type = "top";
        if (!t -> from_res){continue;}
        if (!this -> Zprime.size()){continue;}
        this -> Zprime[0] -> type = "zprime";
        this -> Zprime[0] -> register_child(this -> Tops[x]);
        t -> register_parent(this -> Zprime[0]);  
    }
    
    std::map<std::string, parton*>::iterator itp = this -> m_partons.begin(); 
    for (; itp != this -> m_partons.end(); ++itp){
        parton* px = itp -> second; 
        if (px -> jet_index > -1 && jets.count(px -> jet_index)){
            jet* jt = jets[px -> jet_index]; 
            jt -> register_parent(px);
            px -> jets[jt -> hash] = jt; 
        }

        if (px -> truthjet_index > -1 && tjets.count(px -> truthjet_index)){
            truthjet* jx = tjets[px -> truthjet_index]; 
            jx -> register_parent(px);
            px -> truthjets[jx -> hash] = jx; 
            jx -> top_index = px -> top_index; 
        }

        if (px -> muon_index > -1 && mux.count(px -> muon_index)){
            muon* mx = mux[px -> muon_index]; 
            mx -> register_parent(px);
            px -> register_child(mx); 
        }

        if (px -> electron_index > -1 && elx.count(px -> electron_index)){
            electron* ex = elx[px -> electron_index]; 
            ex -> register_parent(px);
            px -> register_child(ex); 
        }
    }

    this -> broken_event += this -> match_object(&tops_, &this -> m_partons); 
    this -> broken_event  = this -> match_object(&tops_, &this -> m_electrons); 
    this -> broken_event += this -> match_object(&tops_, &this -> m_muons); 
    this -> broken_event += this -> match_object(&tops_, &this -> m_jets); 
    this -> broken_event += this -> match_object(&tops_, &this -> m_truthjets); 

    this -> vectorize(&this -> m_jets     , &this -> Detector); 
    this -> vectorize(&this -> m_muons    , &this -> Detector); 
    this -> vectorize(&this -> m_electrons, &this -> Detector); 

}

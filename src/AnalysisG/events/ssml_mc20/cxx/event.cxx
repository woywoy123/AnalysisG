#include "event.h"

ssml_mc20::ssml_mc20(){
    this -> name = "ssml_mc20"; 
    this -> trees = {"reco"}; 

    this -> add_leaf("eventNumber"   , "eventNumber"); 
    this -> add_leaf("event_category", "event_BkgCategory"); 


    std::string sys = "_NOSYS"; 

    this -> add_leaf("weight"        , "weight_mc"                           + sys); 
    this -> add_leaf("weight_pileup" , "weight_pileup"                       + sys); 
    this -> add_leaf("weight_jvt_sf" , "weight_jvt_effSF"                    + sys); 
    this -> add_leaf("globtrigger_sf", "globalTriggerEffSF"                  + sys); 
    this -> add_leaf("weight_lep_tsf", "weight_leptonSF_tight"               + sys);
    this -> add_leaf("weight_ftag_sf", "weight_ftag_effSF_GN2v01_Continuous" + sys);

    this -> add_leaf("HT_all"   , "HT_all"    + sys);
    this -> add_leaf("ssee"     , "pass_SSee" + sys);   
    this -> add_leaf("ssem"     , "pass_SSem" + sys);   
    this -> add_leaf("ssmm"     , "pass_SSmm" + sys);   
    this -> add_leaf("eem_zveto", "pass_eem_ZVeto" + sys);   
    this -> add_leaf("eee_zveto", "pass_eee_ZVeto" + sys);   
    this -> add_leaf("emm_zveto", "pass_emm_ZVeto" + sys);   
    this -> add_leaf("mmm_zveto", "pass_mmm_ZVeto" + sys);   
    this -> add_leaf("llll_zveto", "pass_llll_ZVeto" + sys);   

    this -> add_leaf("eem", "pass_eem" + sys);   
    this -> add_leaf("eee", "pass_eee" + sys);   
    this -> add_leaf("emm", "pass_emm" + sys);   
    this -> add_leaf("mmm", "pass_mmm" + sys);   

    this -> add_leaf("phi"      , "met_phi"   + sys); 
    this -> add_leaf("met"      , "met_met"   + sys); 
    this -> add_leaf("met_sum"  , "met_sumet" + sys);


    this -> add_leaf("weight_beamsp" , "weight_beamspot"); 

    this -> add_leaf("nElectrons"); 
    this -> add_leaf("nFJets");
    this -> add_leaf("nJets"); 
    this -> add_leaf("nLeptons"); 
    this -> add_leaf("nMuons"); 

    this -> register_particle(&this -> m_zprime); 
    this -> register_particle(&this -> m_tops);
    this -> register_particle(&this -> m_truthjets); 
    this -> register_particle(&this -> m_partons); 
    this -> register_particle(&this -> m_leptons);

    this -> register_particle(&this -> m_electrons); 
    this -> register_particle(&this -> m_muons); 
    this -> register_particle(&this -> m_jets); 
}

ssml_mc20::~ssml_mc20(){}
event_template* ssml_mc20::clone(){return (event_template*)new ssml_mc20();}

void ssml_mc20::build(element_t* el){
    el -> get("eventNumber"   , &this -> eventNumber); 
    el -> get("event_category", &this -> event_category); 

    el -> get("weight"        , &this -> weight_mc); 
    el -> get("weight_beamsp" , &this -> weight_beamspot);
    el -> get("weight_pileup" , &this -> weight_pileup); 
    el -> get("weight_jvt_sf" , &this -> weight_jvt_effSF); 
    el -> get("weight_lep_tsf", &this -> weight_lep_tightSF); 
    el -> get("weight_ftag_sf", &this -> weight_ftag_effSF); 

    el -> get("nElectrons", &this -> n_electrons); 
    el -> get("nFJets"    , &this -> n_fjets); 
    el -> get("nJets"     , &this -> n_jets); 
    el -> get("nLeptons"  , &this -> n_leptons); 
    el -> get("nMuons"    , &this -> n_muons); 

    el -> get("met"       , &this -> met); 
    el -> get("phi"       , &this -> phi); 
    el -> get("met_sum"   , &this -> met_sum); 

    el -> get("globtrigger_sf", &this -> global_trigger_SF); 
    el -> get("HT_all"        , &this -> HT_all);

    char tmp; 
    el -> get("ssee", &tmp); 
    this -> pass_ssee = (int)tmp; 
    el -> get("ssem", &tmp); 
    this -> pass_ssem = (int)tmp; 
    el -> get("ssmm", &tmp); 
    this -> pass_ssmm = (int)tmp; 

    el -> get("eem_zveto", &tmp); 
    this -> pass_eem_zveto = (int)tmp; 
    el -> get("eee_zveto", &tmp); 
    this -> pass_eee_zveto = (int)tmp; 

    el -> get("emm_zveto", &tmp); 
    this -> pass_emm_zveto = (int)tmp; 
    el -> get("mmm_zveto", &tmp); 
    this -> pass_mmm_zveto = (int)tmp; 

    el -> get("eem", &tmp); 
    this -> pass_eem = (int)tmp; 
    el -> get("eee", &tmp); 
    this -> pass_eee = (int)tmp; 

    el -> get("emm", &tmp); 
    this -> pass_emm = (int)tmp; 
    el -> get("mmm", &tmp); 
    this -> pass_mmm = (int)tmp; 

    el -> get("llll_zveto", &tmp); 
    this -> pass_llll_zveto = (int)tmp; 

}

void ssml_mc20::CompileEvent(){
    std::map<int, top*> tops_      = this -> sort_by_index(&this -> m_tops); 
    std::map<int, jet*> jets       = this -> sort_by_index(&this -> m_jets);
    std::map<int, muon*> mux       = this -> sort_by_index(&this -> m_muons); 
    std::map<int, electron*> elx   = this -> sort_by_index(&this -> m_electrons); 
    std::map<int, truthjet*> tjets = this -> sort_by_index(&this -> m_truthjets);
    std::map<int, lepton*>  leps_  = this -> sort_by_index(&this -> m_leptons); 

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
    this -> broken_event += this -> match_object(&tops_, &this -> m_electrons); 
    this -> broken_event += this -> match_object(&tops_, &this -> m_muons); 
    this -> broken_event += this -> match_object(&tops_, &this -> m_jets); 
    this -> broken_event += this -> match_object(&tops_, &this -> m_truthjets); 

    this -> vectorize(&this -> m_jets     , &this -> Detector); 
    this -> vectorize(&this -> m_muons    , &this -> Detector); 
    this -> vectorize(&this -> m_electrons, &this -> Detector); 
    this -> vectorize(&leps_, &this -> Leptsn);
    this -> vectorize(&elx  , &this -> Electrons); 
}

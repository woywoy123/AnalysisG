#include "event.h"

exp_mc20::exp_mc20(){
    this -> name = "experimental_mc20"; 
    this -> add_leaf("met_sum", "met_sumet"); 
    this -> add_leaf("met", "met_met"); 
    this -> add_leaf("phi", "met_phi");
    this -> add_leaf("weight", "weight_mc"); 
    this -> add_leaf("mu", "mu"); 
    this -> add_leaf("event_number", "eventNumber"); 
    this -> trees = {"nominal_Loose"}; 

    this -> register_particle(&this -> m_tops);
    this -> register_particle(&this -> m_children); 
    this -> register_particle(&this -> m_physdet); 
    this -> register_particle(&this -> m_phystru); 
    this -> register_particle(&this -> m_electrons); 
    this -> register_particle(&this -> m_muons); 
    this -> register_particle(&this -> m_jets); 
}

exp_mc20::~exp_mc20(){}

event_template* exp_mc20::clone(){return (event_template*)new exp_mc20();}

void exp_mc20::build(element_t* el){
    el -> get("event_number", &this -> event_number); 
    el -> get("weight", (float*)&this -> weight); 
    el -> get("met_sum", &this -> met_sum); 
    el -> get("met", &this -> met); 
    el -> get("phi", &this -> phi); 
    el -> get("mu", &this -> mu); 
}

void exp_mc20::CompileEvent(){
    std::vector<child*> children_; 
    std::vector<physics_detector*> physdet; 
    std::vector<physics_truth*> phystru; 

    std::map<int, top*> tops_ = this -> sort_by_index(&this -> m_tops); 
    this -> vectorize(&this -> m_children, &children_); 
    this -> vectorize(&this -> m_physdet, &physdet); 
    this -> vectorize(&this -> m_phystru, &phystru); 

    for (size_t x(0); x < children_.size(); ++x){
        child* c = children_[x]; 
        c -> register_parent(tops_[c -> index]); 
        tops_[c -> index] -> register_child(c); 
    }

    for (size_t x(0); x < phystru.size(); ++x){
        physics_truth* c = phystru[x]; 
        for (size_t t(0); t < c -> top_index.size(); ++t){
            if (c -> top_index[t] == -1){continue;}
            c -> register_parent(tops_[t]); 
        }
    } 

    for (size_t x(0); x < physdet.size(); ++x){
        physics_detector* c = physdet[x]; 
        for (size_t t(0); t < c -> top_index.size(); ++t){
            if (c -> top_index[t] == -1){continue;}
            c -> register_parent(tops_[t]); 
        }
    } 


    std::vector<particle_template*> detectors = {}; 
    this -> vectorize(&this -> m_electrons, &detectors); 
    this -> vectorize(&this -> m_muons, &detectors); 
    this -> vectorize(&this -> m_jets, &detectors); 
    for (size_t x(0); x < detectors.size(); ++x){
        std::map<double, physics_detector*>::iterator itr;  
        particle_template* prt = detectors[x]; 

        std::map<double, physics_detector*> maps = {}; 
        for (size_t f(0); f < physdet.size(); ++f){maps[prt -> DeltaR(physdet[f])] = physdet[f];}
        if (!maps.size()){continue;}
        itr = maps.begin();
        if (itr -> first > 0.0001){continue;}

        std::map<std::string, particle_template*> prnt = itr -> second -> parents; 
        if (!prnt.size()){continue;}
        std::map<std::string, particle_template*>::iterator itr_ = prnt.begin(); 
        for (; itr_ != prnt.end(); ++itr_){prt -> register_parent(itr_ -> second);}
    }

    this -> vectorize(&this -> m_tops, &this -> Tops); 
    this -> vectorize(&this -> m_children, &this -> TruthChildren); 
    this -> vectorize(&this -> m_phystru, &this -> PhysicsTruth); 
    this -> vectorize(&this -> m_jets, &this -> Jets); 
    this -> vectorize(&this -> m_electrons, &this -> Leptons); 
    this -> vectorize(&this -> m_muons, &this -> Leptons);
    this -> vectorize(&this -> m_physdet, &this -> PhysicsDetector); 
    this -> Detector = detectors; 
}

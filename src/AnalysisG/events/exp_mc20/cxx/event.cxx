#include <exp_mc20/event.h>

exp_mc20::exp_mc20(){
    this -> name = "experimental_mc20"; 
    this -> add_leaf("met_sum"     , "met_sumet"); 
    this -> add_leaf("met"         , "met_met"); 
    this -> add_leaf("phi"         , "met_phi");
    this -> add_leaf("weight"      , "weight_mc"); 
    this -> add_leaf("mu"          , "mu"); 
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

    float w = 0;
    el -> get("weight", &w); 
    this -> weight = w; 

    el -> get("met_sum", &this -> met_sum); 
    el -> get("met"    , &this -> met); 
    el -> get("phi"    , &this -> phi); 
    el -> get("mu"     , &this -> mu); 
}

void exp_mc20::CompileEvent(){
    auto lamb = [](physics_truth* msp){
        int pdg(0), mx(0); 
        std::map<int, int> out;
        std::map<int, int>::iterator itx; 
        for (size_t x(0); x < msp -> partons.size(); ++x){out[msp -> partons[x]] += 1;}
        for (itx = out.begin(); itx != out.end(); ++itx){
            if (itx -> second < mx){continue;}
            if (std::abs(itx -> first) == 6){continue;}
            if (std::abs(itx -> first) == 24){continue;}
            mx = itx -> second; pdg = itx -> first; 
        }
        return pdg; 
    }; 


    std::map<int, top*> tops_ = this -> sort_by_index(&this -> m_tops); 

    std::vector<child*> children = {}; 
    std::vector<physics_truth*> phystru = {}; 
    std::vector<physics_detector*> physdet = {}; 

    this -> vectorize(&this -> m_children, &children); 
    this -> vectorize(&this -> m_phystru , &phystru); 
    this -> vectorize(&this -> m_physdet , &physdet); 

    for (size_t x(0); x < children.size(); ++x){
        child* c = children[x]; 
        if (!tops_.count(c -> index)){continue;}
        c -> register_parent(tops_[c -> index]); 
        tops_[c -> index] -> register_child(c); 
    }

    for (size_t x(0); x < phystru.size(); ++x){
        physics_truth* c = phystru[x]; 
        for (size_t t(0); t < c -> top_index.size(); ++t){
            int ti = c -> top_index[t]; 
            c -> pdgid = lamb(c); 
            if (ti < 0 || c -> pdgid == 0){continue;}
            c -> register_parent(tops_[ti]);
        }
    } 

    for (size_t x(0); x < physdet.size(); ++x){
        physics_detector* c = physdet[x]; 
        for (size_t t(0); t < c -> top_index.size(); ++t){
            int ti = c -> top_index[t]; 
            if (ti < 0){continue;}
            c -> register_parent(tops_[ti]);
        }
    } 

    this -> vectorize(&this -> m_electrons, &this -> Detector); 
    this -> vectorize(&this -> m_muons    , &this -> Detector); 
    this -> vectorize(&this -> m_jets     , &this -> Detector); 
    for (size_t x(0); x < this -> Detector.size(); ++x){
        particle_template* prt = this -> Detector[x]; 
        std::map<double, physics_detector*> maps = {}; 
        for (size_t f(0); f < physdet.size(); ++f){maps[prt -> DeltaR(physdet[f])] = physdet[f];}

        if (maps.size()){continue;}
        std::map<double, physics_detector*>::iterator itx = maps.begin(); 
        if (itx -> first > 0.0001){continue;}

        std::map<std::string, particle_template*> prnt = itx -> second -> parents; 
        std::map<std::string, particle_template*>::iterator itr = prnt.begin(); 
        for (; itr != prnt.end(); ++itr){prt -> register_parent(itr -> second);}
    }

    this -> vectorize(&this -> m_tops     , &this -> Tops); 
    this -> vectorize(&this -> m_children , &this -> TopChildren); 
    this -> vectorize(&this -> m_phystru  , &this -> PhysicsTruth); 
    this -> vectorize(&this -> m_physdet  , &this -> PhysicsDetector); 

    this -> vectorize(&this -> m_jets     , &this -> Jets); 
    this -> vectorize(&this -> m_electrons, &this -> Leptons); 
    this -> vectorize(&this -> m_muons    , &this -> Leptons);
}

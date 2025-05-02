#include <bsm_4tops/graph_features.h>
#include <bsm_4tops/event.h>

// ------------------ Truth Graph Features --------------------- //
void signal_event(bool* o, bsm_4tops* event){
    if (event -> Tops.size() != 4){*o = false; return;}
    std::vector<particle_template*> t = event -> Tops;
    for (size_t x(0); x < t.size(); ++x){
        top* t_ = (top*)t[x]; 
        if (!t_ -> from_res){continue;}
        *o = true; 
        return;
    }
    *o = false; 
}; 

void num_tops(int* o, bsm_4tops* event){
    *o = event -> Tops.size();
    if (*o > 4){*o = 4;}
}; 

void num_neutrino(int* o, bsm_4tops* event){
    int neutrinos = 0; 
    std::vector<particle_template*> ch = event -> Children; 
    for (size_t x(0); x < ch.size(); ++x){
        top_children* ch_ = (top_children*)ch[x]; 
        if (!ch_ -> is_nu){continue;}
        neutrinos++; 
    }
    *o = neutrinos; 
}; 

void num_lepton(int* o, bsm_4tops* event){
    int leptons = 0; 
    std::vector<particle_template*> ch = event -> Children; 
    for (size_t x(0); x < ch.size(); ++x){
        top_children* ch_ = (top_children*)ch[x]; 
        if (ch_ -> is_nu){continue;}
        if (!ch_ -> is_lep){continue;}
        leptons++; 
    }
    *o = leptons; 
}; 

// --------------------- Graph Observables --------------------- //
void missing_et(double* o, bsm_4tops* event){*o = event -> met;}; 
void missing_phi(double* o, bsm_4tops* event){*o = event -> phi;}; 
void num_jets(double* o, bsm_4tops* event){*o = event -> Jets.size();}
void num_truthjets(double* o, bsm_4tops* event){*o = event -> TruthJets.size();}
void event_number(long* o, bsm_4tops* event){*o = event -> event_number;}
void event_weight(double* o, bsm_4tops* event){*o = event -> weight;}

void num_leps(double* o, bsm_4tops* event){
    *o = int(event -> Muons.size()) + int(event -> Electrons.size());
}

void num_children_leps(double* o, bsm_4tops* event){
    *o = 0;
    std::vector<particle_template*> ch = event -> Children; 
    for (size_t x(0); x < ch.size(); ++x){
        if (ch[x] -> is_nu || !ch[x] -> is_lep){continue;}
        (*o)+=1; 
    }
}


void num_quark(double* o, bsm_4tops* event){
    *o = 0; 
    std::vector<particle_template*> ch = event -> Children; 
    for (size_t x(0); x < ch.size(); ++x){
        if (ch[x] -> is_nu || ch[x] -> is_lep){continue;}
        *o += 1; 
    }
}



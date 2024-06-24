#ifndef GRAPH_FEATURES_BSM_4TOPS_H
#define GRAPH_FEATURES_BSM_4TOPS_H

// ------------------ Truth Graph Features --------------------- //
void static signal_event(bool* o, bsm_4tops* event){
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

void static num_tops(int* o, bsm_4tops* event){
    *o = event -> Tops.size();
    if (*o > 4){*o = 4;}
}; 

void static num_neutrino(int* o, bsm_4tops* event){
    int neutrinos = 0; 
    std::vector<particle_template*> ch = event -> Children; 
    for (size_t x(0); x < ch.size(); ++x){
        top_children* ch_ = (top_children*)ch[x]; 
        if (!ch_ -> is_nu){continue;}
        neutrinos++; 
    }
    *o = neutrinos; 
}; 

void static num_lepton(int* o, bsm_4tops* event){
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
void static missing_et(double* o, bsm_4tops* event){*o = event -> met;}; 
void static missing_phi(double* o, bsm_4tops* event){*o = event -> phi;}; 












#endif

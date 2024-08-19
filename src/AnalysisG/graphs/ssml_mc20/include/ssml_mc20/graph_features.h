#ifndef GRAPH_FEATURES_SSML_MC20_H
#define GRAPH_FEATURES_SSML_MC20_H

// ------------------ Truth Graph Features --------------------- //
void static signal_event(bool* o, ssml_mc20* event){
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

void static num_tops(int* o, ssml_mc20* event){
    *o = event -> Tops.size();
    if (*o > 4){*o = 4;}
}; 

void static num_neutrino(int* o, ssml_mc20* event){
    int neutrinos = 0; 
    std::vector<particle_template*> ch = event -> TruthChildren; 
    for (size_t x(0); x < ch.size(); ++x){
        if (!ch[x] -> is_nu){continue;}
        neutrinos++; 
    }
    *o = neutrinos; 
}; 

void static num_lepton(int* o, ssml_mc20* event){
    int leptons = 0; 
    std::vector<particle_template*> ch = event -> TruthChildren; 
    for (size_t x(0); x < ch.size(); ++x){
        if (ch[x] -> is_nu){continue;}
        if (ch[x] -> is_lep){continue;}
        leptons++; 
    }
    *o = leptons; 
}; 

// --------------------- Graph Observables --------------------- //
void static missing_et(double* o, ssml_mc20* event){*o = event -> met;}; 
void static missing_phi(double* o, ssml_mc20* event){*o = event -> phi;}; 

void static num_children_leps(double* o, ssml_mc20* event){
    *o = 0;
    std::vector<particle_template*> ch = event -> TruthChildren; 
    for (size_t x(0); x < ch.size(); ++x){
        if (ch[x] -> is_nu || !ch[x] -> is_lep){continue;}
        (*o)+=1; 
    }
}

void static num_leps(double* o, ssml_mc20* event){
    *o = int(event -> Leptons.size()); 
}

void static num_quark(double* o, ssml_mc20* event){
    *o = 0; 
    std::vector<particle_template*> ch = event -> TruthChildren; 
    for (size_t x(0); x < ch.size(); ++x){
        if (ch[x] -> is_nu || ch[x] -> is_lep){continue;}
        *o += 1; 
    }
}

void static num_jets(double* o, ssml_mc20* event){*o = event -> Jets.size();}

void static event_number(long* o, ssml_mc20* event){*o = event -> eventNumber;}
void static event_weight(double* o, ssml_mc20* event){*o = event -> weight;}


#endif

#ifndef NODES_FEATURES_SSML_MC20_H
#define NODES_FEATURES_SSML_MC20_H

// --------------------- Node Truth --------------------- //
void static res_node(int* o, particle_template* p){
    std::string type = p -> type;
    if (type == "mu"){*o = ((muon*)p) -> from_res;}
    else if (type == "el"){*o = ((electron*)p) -> from_res;}
    else if (type == "jet"){*o = ((jet*)p) -> from_res;}
    else {*o = 0;} 
}; 

void static top_node(int* o, particle_template* p){
    std::string type = p -> type; 
    if (type == "jet"){
        jet* tx = (jet*)p; 
        if (tx -> top_index < 0){*o = -1;}
        else {*o = tx -> top_index;}
    }

    else if (type == "mu"){
        muon* tx = (muon*)p; 
        *o = tx -> top_index; 
    }

    else if (type == "el"){
        electron* tx = (electron*)p; 
        *o = tx -> top_index; 
    }
    else {*o = -1;} 
}; 

// --------------------- Node Observables --------------------- //
void static pt(double* o, particle_template* p){*o = p -> pt;} 
void static eta(double* o, particle_template* p){*o = p -> eta;} 
void static phi(double* o, particle_template* p){*o = p -> phi;} 
void static energy(double* o, particle_template* p){*o = p -> e;}

void static is_lepton(int* o, particle_template* p){*o = (p -> is_lep && !p -> is_nu);}
void static is_bquark(int* o, particle_template* p){*o = p -> is_b;}
void static is_neutrino(int* o, particle_template* p){*o = p -> is_nu;}


#endif

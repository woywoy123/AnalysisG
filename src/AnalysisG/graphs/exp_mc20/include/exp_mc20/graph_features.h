#ifndef GRAPH_FEATURES_EXP_MC20_H
#define GRAPH_FEATURES_EXP_MC20_H

// ------------------ Truth Graph Features --------------------- //
void static signal_event(bool* o, exp_mc20* event){
    if (event -> Tops.size() != 4){*o = false; return;}
    //std::vector<particle_template*> t = event -> Tops;
    //for (size_t x(0); x < t.size(); ++x){
    //    top* t_ = (top*)t[x]; 
    //    continue; 
    //    //if (f!t_ -> from_res){continue;}
    //    //*o = true; 
    //    //return;
    //}
    *o = false; 
}

void static num_tops(int* o, exp_mc20* event){*o = (event -> Tops.size() >= 4) ? 4 : event -> Tops.size();}

void static num_neutrino(int* o, exp_mc20* event){
    int neutrinos = 0; 
    std::vector<particle_template*> ch = event -> TopChildren; 
    for (size_t x(0); x < ch.size(); ++x){
        if (!ch[x] -> is_nu){continue;}
        neutrinos++; 
    }
    *o = neutrinos; 
}

void static num_lepton(int* o, exp_mc20* event){
    int leptons = 0; 
    std::vector<particle_template*> ch = event -> TopChildren; 
    for (size_t x(0); x < ch.size(); ++x){
        if (ch[x] -> is_nu){continue;}
        if (ch[x] -> is_lep){continue;}
        leptons++; 
    }
    *o = leptons; 
}; 

// --------------------- Graph Observables --------------------- //
void static missing_et(double* o, exp_mc20* event){*o = event -> met;}; 
void static missing_phi(double* o, exp_mc20* event){*o = event -> phi;}; 

void static num_children_leps(double* o, exp_mc20* event){
    double num_leps = 0; 
    std::vector<particle_template*> ch = event -> TopChildren; 
    for (size_t x(0); x < ch.size(); ++x){
        if (ch[x] -> is_nu || !ch[x] -> is_lep){continue;}
        num_leps += 1; 
    }
    *o = num_leps;
}

void static num_leps(double* o, exp_mc20* event){*o = int(event -> Leptons.size());}
void static num_jets(double* o, exp_mc20* event){*o = event -> Jets.size();}
void static event_number(long* o, exp_mc20* event){*o = event -> event_number;}
void static event_weight(double* o, exp_mc20* event){*o = event -> weight;}


#endif

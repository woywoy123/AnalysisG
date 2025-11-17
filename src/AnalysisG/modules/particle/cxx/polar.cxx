#include <templates/particle_template.h>

void particle_template::set_pt(double* val, particle_template* prt){ 
    prt -> data.pt = *val; 
    prt -> data.cartesian = true;
}

void particle_template::get_pt(double* val, particle_template* prt){
    prt -> to_polar(); 
    *val = prt -> data.pt;
}


void particle_template::set_eta(double* val, particle_template* prt){ 
    prt -> data.eta = *val; 
    prt -> data.cartesian = true; 
}

void particle_template::get_eta(double* val, particle_template* prt){
    prt -> to_polar(); 
    *val = prt -> data.eta;
}

void particle_template::set_phi(double* val, particle_template* prt){ 
    prt -> data.phi = *val; 
    prt -> data.cartesian = true; 
}


void particle_template::get_phi(double* val, particle_template* prt){
    prt -> to_polar(); 
    *val = prt -> data.phi;
}

void particle_template::to_polar(){
    particle_t* p = &this -> data; 
    if (!p -> polar){ return; }

    // Transverse Momenta
    p -> pt  = std::pow(p -> px, 2); 
    p -> pt += std::pow(p -> py, 2);
    p -> pt  = std::sqrt(p -> pt); 

    // Rapidity 
    p -> eta = std::asinh(p -> pz / p -> pt); 
    p -> phi = std::atan2(p -> py, p -> px);  
    p -> polar = false; 
}


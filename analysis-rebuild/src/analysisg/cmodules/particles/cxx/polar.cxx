#include <particles/particles.h>

void particles::pt(double val){ 
    this -> data.pt = val; 
    this -> data.cartesian = true;
}

void particles::eta(double val){ 
    this -> data.eta = val; 
    this -> data.cartesian = true; 
}

void particles::phi(double val){ 
    this -> data.phi = val; 
    this -> data.cartesian = true; 
}

double particles::pt(){
    this -> to_polar(); 
    return this -> data.pt;
}

double particles::eta(){
    this -> to_polar(); 
    return this -> data.eta;
}

double particles::phi(){
    this -> to_polar(); 
    return this -> data.phi;
}

void particles::to_polar(){
    particle_t* p = &this -> data; 
    if (!p -> polar){ return; }

    // Transverse Momenta
    p -> pt  = std::pow(p -> px, 2); 
    p -> pt += std::pow(p -> py, 2);
    p -> pt  = std::pow(p -> pt, 0.5); 

    // Rapidity 
    p -> eta = std::asinh(p -> pz / p -> pt); 
    p -> phi = std::atan2(p -> py, p -> px);  
    p -> polar = false; 
}


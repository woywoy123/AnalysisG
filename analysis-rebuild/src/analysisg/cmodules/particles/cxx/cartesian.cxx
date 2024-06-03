#include <particles/particles.h>

void particles::px(double val){ 
    this -> data.px = val; 
    this -> data.polar = true; 
}

void particles::py(double val){ 
    this -> data.py = val; 
    this -> data.polar = true; 
}

void particles::pz(double val){ 
    this -> data.pz = val; 
    this -> data.polar = true; 
}

void particles::to_cartesian(){
    particle_t* p = &this -> data; 
    if (!p -> cartesian){ return; }
    p -> px = (p -> pt)*std::cos(p -> phi); 
    p -> py = (p -> pt)*std::sin(p -> phi); 
    p -> pz = (p -> pt)*std::sinh(p -> eta); 
    p -> cartesian = false; 
}

double particles::px(){
    this -> to_cartesian(); 
    return this -> data.px;
}

double particles::py(){
    this -> to_cartesian(); 
    return this -> data.py;
}

double particles::pz(){
    this -> to_cartesian(); 
    return this -> data.pz;
}


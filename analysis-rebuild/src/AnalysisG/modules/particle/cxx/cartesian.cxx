#include <templates/particle_template.h>

void particle_template::set_px(double* val, particle_template* prt){ 
    prt -> data.px = *val; 
    prt -> data.polar = true; 
}

void particle_template::get_px(double* val, particle_template* prt){
    prt -> to_cartesian(); 
    *val = prt -> data.px;
}

void particle_template::set_py(double* val, particle_template* prt){ 
    prt -> data.py = *val; 
    prt -> data.polar = true; 
}

void particle_template::get_py(double* val, particle_template* prt){
    prt -> to_cartesian(); 
    *val = prt -> data.py;
}

void particle_template::set_pz(double* val, particle_template* prt){ 
    prt -> data.pz = *val; 
    prt -> data.polar = true; 
}

void particle_template::get_pz(double* val, particle_template* prt){
    prt -> to_cartesian(); 
    *val = prt -> data.pz;
}

void particle_template::to_cartesian(){
    particle_t* p = &this -> data; 
    if (!p -> cartesian){ return; }
    p -> px = (p -> pt)*std::cos(p -> phi); 
    p -> py = (p -> pt)*std::sin(p -> phi); 
    p -> pz = (p -> pt)*std::sinh(p -> eta); 
    p -> cartesian = false; 
}



#include <templates/particle_template.h>

double particle_template::DeltaR(particle_template* part){
    double sum = std::fabs( this -> phi - part -> phi);
    sum = std::fmod(sum, 2*M_PI); 
    sum = M_PI - std::fabs(sum - M_PI); 
    sum = std::pow(sum, 2);
    sum += std::pow(this -> eta - part -> eta, 2); 
    return std::pow(sum, 0.5); 
}

void particle_template::get_e(double* v, particle_template* prt) {
    particle_t* p = &prt -> data; 
    if (p -> e >= 0){*v = p -> e; return;}
    p -> e += std::pow(prt -> px, 2); 
    p -> e += std::pow(prt -> py, 2); 
    p -> e += std::pow(prt -> pz, 2); 
    if (p -> mass >= 0){p -> e += p -> mass;}
    p -> e  = std::pow(p -> e, 0.5); 
    *v = p -> e; 
}

void particle_template::set_e(double* val, particle_template* prt){ 
    prt -> data.e = *val; 
}

void particle_template::set_mass(double* val, particle_template* prt){ 
    prt -> data.mass = *val; 
}

void particle_template::get_mass(double* val, particle_template* prt){
    particle_t* p = &prt -> data; 
    if (p -> mass > -1){ *val = p -> mass; return; }
    p -> mass = 0; 
    p -> mass -= std::pow(prt -> px, 2); 
    p -> mass -= std::pow(prt -> py, 2); 
    p -> mass -= std::pow(prt -> pz, 2); 
    p -> mass += std::pow(prt -> e , 2); 
    p -> mass = (p -> mass >= 0) ? std::pow(p -> mass, 0.5) : -1; 
    *val = p -> mass; 
}




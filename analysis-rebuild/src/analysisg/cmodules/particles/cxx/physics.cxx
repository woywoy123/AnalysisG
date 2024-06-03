#include <particles/particles.h>

double particles::DeltaR(particles* part){
    double sum = std::fabs( this -> phi() - part -> phi());
    sum = std::fmod(sum, 2*M_PI); 
    sum = M_PI - std::fabs(sum - M_PI); 
    sum = std::pow(sum, 2);
    sum += std::pow(this -> eta() - part -> eta(), 2); 
    return std::pow(sum, 0.5); 
}

double particles::e() {
    particle_t* p = &this -> data; 
    if (p -> e >= 0){return p -> e;}
    p -> e += std::pow(this -> px(), 2); 
    p -> e += std::pow(this -> py(), 2); 
    p -> e += std::pow(this -> pz(), 2); 
    if (p -> mass >= 0){p -> e += p -> mass;}
    p -> e  = std::pow(p -> e, 0.5); 
    return p -> e; 
}

void particles::e(double val){ 
    this -> data.e = val; 
}

void particles::mass(double val){ 
    this -> data.mass = val; 
}

double particles::mass(){
    particle_t* p = &this -> data; 
    if (p -> mass > -1){ return p -> mass; }
    p -> mass = 0; 
    p -> mass -= std::pow(this -> px(), 2); 
    p -> mass -= std::pow(this -> py(), 2); 
    p -> mass -= std::pow(this -> pz(), 2); 
    p -> mass += std::pow(this -> e() , 2); 
    p -> mass = (p -> mass >= 0) ? std::pow(p -> mass, 0.5) : -1; 
    return p -> mass; 
}




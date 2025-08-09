#include "particle.h"
#include <iostream>

particle::particle(double px, double py, double pz, double e){
    this -> px = px; this -> py = py; 
    this -> pz = pz; this -> e  = e; 
    this -> init(); 
}

particle::particle(double px, double py, double pz){
    this -> px = px; this -> py = py; 
    this -> pz = pz; this -> e  = -1; 
    this -> init(); 
}

particle::particle(const particle& p_){
    this -> px = p_.px; this -> py = p_.py; 
    this -> pz = p_.pz; this -> e  = p_.e; 
    this -> init(); 
}

void particle::init(){
    this -> p2  = this -> px * this -> px; 
    this -> p2 += this -> py * this -> py; 
    this -> p2 += this -> pz * this -> pz; 
    this -> p  = std::pow(this -> p2, 0.5); 
    if (this -> e < 0){this -> e = this -> p;}
    this -> e2 = this -> e * this -> e; 
    this -> m2 = this -> e2 - this -> p2; 

    this -> m  = std::pow(this -> m2, 0.5); 
    this -> b  = this -> p / this -> e; 
    this -> b2 = this -> b * this -> b; 
}


particle particle::operator+(const particle& other){
    return particle(this -> px + other.px, this -> py + other.py, this -> pz + other.pz); 
}

particle particle::operator-(const particle& other){
    return particle(this -> px - other.px, this -> py - other.py, this -> pz - other.pz); 
}

particle particle::operator*(double scalar){
    return particle(this -> px * scalar, this -> py * scalar, this -> pz * scalar);
}

void particle::print(){
    std::cout << "(";
    std::cout << this -> px << ", ";
    std::cout << this -> py << ", ";
    std::cout << this -> pz << ", ";
    std::cout << this -> e << ")" << std::endl;
}


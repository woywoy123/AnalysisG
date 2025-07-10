#include "particle.h"

particle::particle(double px, double py, double pz, double e){
    this -> px = px; this -> py = py;
    this -> pz = pz; this -> e  = e; 
}
double particle::p(){return pow(this -> p2(), 0.5);}
double particle::p2(){return pow(this -> px, 2) + pow(this -> py, 2) + pow(this -> pz, 2);}
double particle::m2(){return pow(this -> e, 2) - this -> p2();}
double particle::m(){return pow(this -> m2(), 0.5);}
double particle::beta(){return this -> p() / this -> e;}
double particle::beta2(){return pow(this -> beta(), 2);}
double particle::phi(){return std::atan2(this -> py, this -> px);}
double particle::theta(){return std::atan2(pow(pow(this -> px, 2) + pow(this -> py, 2), 0.5), this -> pz);}
particle::~particle(){}; 
particle* particle::clone(){return new particle(this -> px, this -> py, this -> pz, this -> e);}




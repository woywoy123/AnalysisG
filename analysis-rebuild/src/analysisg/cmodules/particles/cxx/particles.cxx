#include <particles/particles.h>

particles::particles(){}
particles::particles(particle_t* p){this -> data = *p;}
particles::particles(double px, double py, double pz, double e){
    particle_t* p = &this -> data; 
    p -> px = px; 
    p -> py = py; 
    p -> pz = pz; 
    p -> e = e; 
    p -> polar = true; 
}

particles::particles(double px, double py, double pz){
    particle_t* p = &this -> data; 
    p -> px = px; 
    p -> py = py; 
    p -> pz = pz; 
    this -> e(); 
    p -> polar = true; 
}

particles::~particles(){}

std::string particles::hash(){
    particle_t* p = &this -> data; 
    if ((p -> hash).size()){return p -> hash;}

    this -> to_cartesian(); 
    p -> hash  = tools().to_string(this -> px()); 
    p -> hash += tools().to_string(this -> py()); 
    p -> hash += tools().to_string(this -> pz());
    p -> hash += tools().to_string(this -> e()); 
    p -> hash  = tools().hash(p -> hash, 18); 
    return p -> hash; 
}

particles* particles::operator + (particles* p){
    p -> to_cartesian(); 
    particles* p2 = new particles(
            p -> px() + this -> px(), p -> py() + this -> py(), 
            p -> pz() + this -> pz(), p -> e()  + this -> e()
    ); 

    p2 -> to_cartesian(); 
    p2 -> to_polar(); 
    p2 -> data.type = this -> data.type; 
    return p2; 
}

void particles::operator += (particles* p){
    p -> to_cartesian(); 
    this -> to_cartesian();
    this -> data.px += p -> px(); 
    this -> data.py += p -> py(); 
    this -> data.pz += p -> pz(); 
    this -> data.e  += p -> e(); 
    this -> data.polar = true;
}

void particles::iadd(particles* p){
    *this += p; 
}

bool particles::operator == (particles& p){
    return this -> hash() == p.hash(); 
}

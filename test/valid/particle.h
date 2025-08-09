#ifndef PARTICLE_H
#define PARTICLE_H
#include <math.h>

struct particle {
    particle(double px, double py, double pz, double e);
    particle(double px, double py, double pz); 
    particle(const particle& p);
    
    double m2 = 0, p2 = 0, b2 = 0, e2 = 0; 
    double  m = 0,  p = 0,  b = 0,  e = 0; 
    double px = 0, py = 0, pz = 0;
    
    particle operator+(const particle& other); 
    particle operator-(const particle& other); 
    particle operator*(double scalar); 
    void print(); 
    void init();
};

#endif

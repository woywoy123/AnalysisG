#ifndef H_PARTICLE
#define H_PARTICLE

#include <iostream>
#include <iomanip>
#include <cmath>

class particle {
    public:
        particle(double px, double py, double pz, double e); 
        ~particle();
        double p(); 
        double p2(); 
        double m(); 
        double m2(); 
        double beta();
        double beta2(); 
        double phi(); 
        double theta(); 

        double px = 0; 
        double py = 0;
        double pz = 0;
        double e  = 0; 
        particle* clone(); 
}; 

#endif

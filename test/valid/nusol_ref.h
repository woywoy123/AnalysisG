#ifndef REFERENCE_NUSOL_H
#define REFERENCE_NUSOL_H

#include "structs.h"
#include "particle.h"

struct reference_t {
    double x0p = 0; 
    double x0  = 0;
    double x1  = 0;
    double y1  = 0; 
    double Sx  = 0; 
    double Sy  = 0;
    double Z2  = 0;  
    double Z = 0;
    matrix HT; 
    matrix H; 
    
    vec3 nu(double phi);
}; 

class nusol
{
    public:
        nusol(const particle& b, const particle& l); 
        ~nusol(); 

        reference_t update(double mT_, double mW_); 
        void make_rt(); 

    private:
        particle*  bjet = nullptr; 
        particle*  lep  = nullptr; 
        matrix*    rt  = nullptr; 

        double mW2 = -1;
        double mT2 = -1; 

}; 

#endif

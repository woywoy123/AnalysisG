#ifndef H_CONUIC
#define H_CONUIC
#include <conuix/struct.h>
#include <structs/property.h>
#include <complex>
#include <math.h>

class particle_template; 


class conuic {
    
    public:
        conuic(particle_template* jet, particle_template* lep); 
        ~conuic(); 

        cproperty<long double, conuic> t;  // tau
        cproperty<long double, conuic> z;  // scale
        cproperty<long double, conuic> l;  // lambda
         
        matrix_t get_RT(); 
        long double Z2(long double Sx, long double Sy);  
        long double Sx(long double Tau, long double Z);
        long double Sy(long double Tau, long double Z); 

        void debug(); 
        void solve();  

    private:
        atomics_t* cache = nullptr;
        particle_template* _jet = nullptr; 
        particle_template* _lep = nullptr; 

        void static set_tau(long double* v, conuic* c); 
        void static set_scl(long double* v, conuic* c); 
        void static set_lmb(long double* v, conuic* c); 

        void static get_tau(long double* v, conuic* c); 
        void static get_scl(long double* v, conuic* c); 
        void static get_lmb(long double* v, conuic* c); 

        long double tau  = 0; // simple hyperbolic parameter
        long double stau = 0; // pre-compute sinh
        long double ctau = 0; // pre-compute cosh
        long double ttau = 0; // pre-compute tanh

        long double scale = 0;
        long double lamb  = 0; 

}; 


#endif

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

        long double tstar = 0;
        long double error = 0;
        long double theta = 0; 
        bool    converged = false;
        matrix_t vstar; 

        long double Z2(long double Sx, long double Sy);  
        long double Sx(long double Tau, long double Z);
        long double Sy(long double Tau, long double Z); 
        long double x1(long double t, long double Z); 
        long double y1(long double t, long double Z); 
        bool get_TauZ(long double sx, long double sy, long double* z, long double* t); 

        long double P(long double l, long double t, long double Z); 
        long double dPdt(long double l, long double t, long double Z); 
        long double dPdtL0(long double t, long double Z); 
        long double dPl0(long double t); 
        matrix_t Hmatrix(long double t, long double Z); 

        void debug(); 

    private:
        atomics_t* cache = nullptr;
        particle_template* _jet = nullptr; 
        particle_template* _lep = nullptr; 
}; 


#endif

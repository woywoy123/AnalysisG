#ifndef H_BASE
#define H_BASE
#include "particle.h"
#include "matrix.h"

class nusol {
    public:
        nusol(particle* b, particle* l, double mW, double mT); 
        double Sx(); 
        double dSx_dmW();

        double Sy(); 
        double dSy_dmW(); 
        double dSy_dmT(); 

        double w();
        double w2(); 
        double om2(); 

        double Z(); 
        double Z2(); 
        double dZ_dmT(); 
        double dZ_dmW(); 

        double  x1();
        double dx1_dmW(); 
        double dx1_dmT(); 

        double  y1(); 
        double dy1_dmW(); 
        double dy1_dmT(); 

        void get_mw(double* v_crit, double* v_infl); 
        void get_mt(double* v_crit, double* v_infl); 

        double** N(); 
        double** H(); 
        double** H_perp(); 
        double** H_tilde();
        double** dH_dmW();
        double** dH_dmT(); 

        double** R_T();
        void Z2_coeff(double* A, double* B, double* C); 
        ~nusol();  

    private:
        particle* b = nullptr;
        particle* l = nullptr; 
        double mw = 0;
        double mt = 0; 
        double _s = 0; 
        double _c = 0; 

        double** h = nullptr; 
        double** r_t = nullptr;  
        double** dw_H = nullptr;
        double** dt_H = nullptr;
        double** h_tilde = nullptr;
        double** h_perp  = nullptr; 
        double** n_matrx = nullptr; 

}; 

#endif

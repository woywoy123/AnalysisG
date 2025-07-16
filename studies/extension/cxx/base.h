#ifndef H_BASE
#define H_BASE
#include "particle.h"
#include "matrix.h"
#include "mtx.h"

class nusol {
    public:
        nusol(particle* b, particle* l, double mW, double mT, bool delP); 
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

        void r_mW(double* mw1, double* mw2); 
        void get_mw(double* v_crit, double* v_infl); 
        void get_mt(double* v_crit, double* v_infl); 
        void update(double mt, double mw);
        void flush(); 

        mtx* N(); 
        mtx* H(); 
        mtx* H_perp(); 
        mtx* H_tilde();
        mtx* dH_dmW();
        mtx* dH_dmT(); 
        mtx* K(); 
        mtx* R_T();

        void Z2_coeff(double* A, double* B, double* C); 

        void misc(); 
        ~nusol();  

    private:
        particle* b = nullptr;
        particle* l = nullptr; 

        double mw = 0;
        double mt = 0; 
        double _s = 0; 
        double _c = 0; 
        bool delp = true; 

        mtx* h_tilde = nullptr; 
        mtx* h       = nullptr; 
        mtx* r_t     = nullptr; 
        mtx* dw_H    = nullptr; 
        mtx* dt_H    = nullptr; 
        mtx* h_perp  = nullptr; 
        mtx* n_matrx = nullptr; 
        mtx* k       = nullptr; 
}; 

#endif

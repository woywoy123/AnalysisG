#ifndef H_MULTISOL_CONUIX
#define H_MULTISOL_CONUIX
#include "multisol/cache.h"

struct params_t {
    long double mass_t = 172.1 * 1000; 
    long double mass_w = 81.381 * 1000; 

    long double met = 0; 
    long double phi = 0; 

    long double metx = 0; 
    long double mety = 0; 

    long double violation = 0.001; 
    long double tstar_lim = 0.00001; 

    std::vector<particle_template*>* targets;
}; 

class conuic : public cache {

    public:
        conuic(particle_template* jt, particle_template* lep);
        ~conuic(); 

        long double Z2(long double sx, long double sy); 
        
        long double Sx(long double ct, long double st, long double Z); 
        long double Sy(long double ct, long double st, long double Z); 

        long double Sx(long double mw, long double mt); 
        long double Sy(long double mw, long double mt); 

        long double x1(long double ct, long double st, long double Z); 
        long double y1(long double ct, long double st, long double Z); 

        matrix_t H_tilde(long double ct, long double st);
        matrix_t H_matrix(long double ct, long double st);
        matrix_t H_perp(long double ct, long double st, long double Z);
        matrix_t N(long double ct, long double st, long double Z, bool full); 
        matrix_t K(long double ct, long double st, long double Z, bool full); 
        matrix_t Nu(const matrix_t nu, long double Z, bool full); 

        coef_t masses(long double Z, long double t); 
        coef_t mass_line(long double mW, long double mT); 

        coef_t root_mW(long double mT); 
        coef_t get_tauZ(long double sx, long double sy);


        // ---- hyperbolic characteristic polyomial ----- //
        long double dPl0(); 
        long double PL(long double z, long double l); 
        long double dPdt(long double z, long double l); 
        long double dPdtL0(long double z); 
        coef_t dPdZ0(long double z);

        void intersection(
            conuic* nux, long double metx, long double mety, 
            long double t1, long double z1, 
            long double t2, long double z2
        ); 

        cache_t state;   
}; 





#endif

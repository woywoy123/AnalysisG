#ifndef H_MULTISOL_CACHE
#define H_MULTISOL_CACHE
#include <templates/particle_template.h>
#include "multisol/matrix.h"
#include <string.h>

struct coef_t {
    long double a = 0;
    long double b = 0;
    long double c = 0;
    long double d = 0;
    long double e = 0;

    std::complex<long double> a_cplx; 
    std::complex<long double> b_cplx; 
}; 

struct cache_t {
    const matrix_t* RT = nullptr; 

    const matrix_t* HBX = nullptr; 
    const matrix_t* HBC = nullptr;
    const matrix_t* HBS = nullptr; 

    const matrix_t* HTX = nullptr; 
    const matrix_t* HTC = nullptr;
    const matrix_t* HTS = nullptr; 

    coef_t Z2;
    coef_t Sx;
    coef_t Sy; 

    coef_t x1;
    coef_t y1; 

    coef_t tZ; 
    coef_t mass; 
    coef_t line; 

    coef_t lep;
    coef_t jet;  

    coef_t r_mW; 
}; 



class cache {

    public:
        cache(particle_template* jt, particle_template* lp);
        ~cache(); 

        cache_t init(); 
        
        void hyper(long double tau); 
        long double alpha_p(long double u); // a + b * tanh(tau)
        long double alpha_m(long double u); // c + d * tanh(tau)
        // a: Omega * tpsi, b:          beta_mu
        // c: Omega       , d: - tpsi * beta_mu 

        long double _tt = 0; 
        long double _ct = 0;
        long double _st = 0; 

        long double _cpsi = 0; 
        long double _spsi = 0; 
        long double _tpsi = 0; 

        long double _sth = 0; 
        long double _cth = 0; 
        long double _o   = 0; 

        // -------- characteristics --------- //
        long double midpoint; 
        long double poles[2]; 
        long double mobius[4]; 
        long double taustar[4]; 

        long double taupts[100]; 

        long double beta = 0; 
        long double a_   = 0; 

        bool converged = false; 

        std::complex<long double> sym_axis; 
        std::complex<long double> kfactor; 
        std::complex<long double> e_val[2]; 
        std::complex<long double> e_vec[4]; 
        std::complex<long double> fixed[2];  

        matrix_t make_w(const matrix_t* v); 
        matrix_t make_top(const matrix_t* v);
        matrix_t make_neutrino(const matrix_t* v); 

        std::complex<long double> get_mass(const matrix_t* v); 
        std::string hash = ""; 

    private:
        template <typename g>
        void dSafe(g** v){
            if (*v){return;}
            delete *v; *v = nullptr; 
        }

        matrix_t* NU = nullptr; 
        matrix_t* RT = nullptr; 

        matrix_t* HBX = nullptr; 
        matrix_t* HBC = nullptr;
        matrix_t* HBS = nullptr; 

        matrix_t* HTX = nullptr; 
        matrix_t* HTC = nullptr;
        matrix_t* HTS = nullptr; 

        matrix_t* vlep = nullptr; 
        matrix_t* vjet = nullptr; 

        void get_hmatrix(
            long double tpsi, long double cpsi, long double spsi, 
            long double    o, long double lb
        ); 

        const matrix_t* get_rotation(
            long double lpx, long double lpy, long double lpz, long double lp,
            long double bpx, long double bpy, long double bpz
        );                

        void get_dPL0dT0();

        particle_template* jet = nullptr; 
        particle_template* lep = nullptr; 

}; 

#endif

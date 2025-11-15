#ifndef H_CONUIX_STRUCT
#define H_CONUIX_STRUCT
#include <conuix/matrix.h>
#include <conuix/htilde.h>
#include <iostream>
#include <iomanip>
#include <string>

class particle_template; 
struct atomics_t; 

namespace Conuix {
    struct debug {
        virtual void print(int p = 16);
        void variable(std::string name, long double val); 
        int prec = 16; 
    }; 

    struct kinematic_t : public debug {
        void print(int p = 16) override; 
        long double beta = 0;
        long double mass = 0;
        long double energy = 0; 
    }; 

    struct rotation_t : public debug {
        void print(int p = 16) override; 
        long double phi   = 0;
        long double theta = 0;
        matrix_t vec;
        matrix_t R_T; 
    }; 
    
    struct base_t : public debug {
        void print(int p = 16) override; 

        long double rbl = 0;

        long double cos = 0;
        long double sin = 0;

        long double w   = 0;
        long double o   = 0;

        long double w2  = 0;
        long double o2  = 0; 

        long double beta = 0; 
        long double mass = 0;
        long double E    = 0;

        long double tpsi = 0;
        long double cpsi = 0;
        long double spsi = 0; 
    };

    struct thetapsi_t {
        long double p_sin = 0; 
        long double m_sin = 0;
        long double p_cos = 0;
        long double m_cos = 0; 
    }; 

    struct pencil_t : debug {
        long double Z2(long double Sx, long double Sy);
        void print(int p) override; 

        long double a = 0;
        long double b = 0;
        long double c = 0;
        long double d = 0;
        long double e = 0;
    }; 

    struct Sx_t : debug {
        long double Sx(long double tau, long double Z);
        void print(int p) override; 

        long double a = 0;
        long double b = 0;
        long double c = 0; 
    };

    struct Sy_t : debug {
        long double Sy(long double tau, long double Z);
        void print(int p) override; 

        long double a = 0;
        long double b = 0;
        long double c = 0; 
    };

    struct H_matrix_t : debug {
        matrix_t H_Matrix(long double tau, long double Z);
        matrix_t H_Tilde(long double tau, long double Z); 
        void print(int p) override; 

        // ------ H-bar -----
        matrix_t HBX;
        matrix_t HBS;
        matrix_t HBC; 

        // ------ H_matrix ----- //
        matrix_t HTX;
        matrix_t HTS;
        matrix_t HTC;
    };

    struct Mobius_t : debug {
        void init(base_t* base);  

        long double alpha_p(long double u); 
        long double alpha_m(long double u); 
        // a: Omega * tpsi, b:          beta_mu
        // c: Omega       , d: - tpsi * beta_mu 
        long double a, b, c, d;

        long double dPl0(long double x, bool use_tanh = false); 
        long double a_; // beta * cos(psi)**3

        std::complex<long double> eig_v1; 
        std::complex<long double> eig_v2; 
        
        std::complex<long double> eig_vx[2]; 
        std::complex<long double> eig_vy[2]; 

        std::complex<long double> kfactor; 
        
        std::complex<long double> f1_pts; 
        std::complex<long double> f2_pts; 
    
        std::complex<long double> mid; 



    }; 






    long double cos_theta(particle_template* jet, particle_template* lep); 
}


struct atomics_t : public Conuix::debug {
    atomics_t(particle_template* jet, particle_template* lep, double m_nu = 0); 
    ~atomics_t(); 

    long double x1(long double Z, long double t); 
    long double y1(long double Z, long double t); 
    bool GetTauZ(long double sx, long double sy, long double* z, long double* t); 
    void masses(long double Z, long double t, std::complex<long double>* mw, std::complex<long double>* mt); 
    void debug_mode(particle_template* jet, particle_template* lep); 

    // ---- Kinematics of the jet and lepton pairs ---- // 
    Conuix::kinematic_t lp; // === Lepton
    Conuix::kinematic_t jt; // === jet
    Conuix::kinematic_t nu; // === neutrino

    // ---- Get base kinematic variables ---- //
    Conuix::base_t     base;
    Conuix::rotation_t rotation;
    
    // ----------- mapping from psi to theta -------- //
    Conuix::thetapsi_t psi_theta;
    
    // -------- Pencil function coefficients ------ //
    Conuix::pencil_t pencil;      

    // ---------- Shift parameters -------------- //
    Conuix::Sx_t Sx; 
    Conuix::Sy_t Sy;

    // ---------- H and H_tilde matrices -------- //
    Conuix::H_matrix_t H_Matrix; 

    // ---------- Mobius --------- //
    // Enter at your own risk....
    Conuix::Mobius_t Mobius; 





}; 

#endif



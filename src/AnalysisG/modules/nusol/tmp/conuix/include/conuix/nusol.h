#ifndef H_NUSOL_CNS
#define H_NUSOL_CNS

#include <templates/particle_template.h>
#include <conics/matrix.h>

struct eig_t {
    double r1 = 0, r2 = 0; 
    double i1 = 0, i2 = 0; 
}; 


struct nuclx_t {

    // use this to initialize all the constants.
    nuclx_t(particle_template* jet, particle_template* lep); 
   
    // --------- translation layer ---------- //
    // ****************** //
    nuclx_t from_sx_sy(double _sx, double _sy); 
    double get_t(double _vp, double _up, double _z); 
    double get_z(double _sx, double _sy);
    
    // reference point to hyperbola center
    double u = 0, v = 0; 

    // inverse the rotation and align with hyperbola principal axes
    double u_p = 0, v_p = 0; 

    // corresponding parameterizations 
    double t_v = 0, z_v = 0;  // ---- z < 0 means failed: non-physical
    // ******************* //

    // ****************** //
    nuclx_t from_z_t(double _z, double _t); 
    double get_mt(double _sx, double _sy); 
    double get_mw(double _sx);
    
    // sx and sy values from t, z
    double sx_v = 0, sy_v = 0; 

    // top and w-boson mass from t, z.
    double mt_v = 0, mw_v = 0; 
    // ******************* //


    // --------- constants ------- //
    void surface(); 
    double A = 0, B = 0, C = 0; 
    double D = 0, E = 0, F = 0; 
   
    // shift of surface 
    void shifts(); 
    double s0x = 0, s0y = 0; 

    // rotation angle
    double psi = 0, cpsi = 0, spsi = 0; 

    // lambdas - eigenvalues
    double lmb1 = -1, lmb2 = 0; 

    // parameterization
    void sx(); 
    double a_x = 0, b_x = 0, c_x = 0;  // Sx
                                      
    void sy(); 
    double a_y = 0, b_y = 0, c_y = 0;  // Sy 

    // mass parameterization
    void mw(); 
    double a_w = 0, b_w = 0;  // mW
    
    void mt();
    double a_t = 0, b_t = 0, c_t = 0;  // mT

    // ******** eigenvalue analysis ******* //
    // characteristic polynomial
    // P(lambda, t, z) = 
    // lambda^3 + 
    // z * a_l lambda^2 + 
    // z^2 * b_l * cosh(t) * lambda  + z^2 * c_l * sinh(t) * lambda + 
    // z^3 * d_l * sinh(t)
    void polynomial(); 
    double a_l = 0, b_l = 0, c_l = 0, d_l = 0; 

    // computes the lower bound of t where dP_dL can be physically 0.
    void critical_t0();
    double t_0 = 0; 

    // ---------- matrix definitions -------- //
    void H_bar(); 
    matrix_t HBc, HB1, HB2; // pre-rotation

    void H(); 
    matrix_t Hc, H1, H2;  // post-rotation

    // decay frame to lab frame rotation.
    void rotation(); 
    matrix_t vec_jet, R_T; 

    // ---------- particle parameters --------- //
    // kinematics  
    double beta_lep = 0, beta_jet = 0; 
    double mass_lep = 0, mass_jet = 0; 
    double p_lep    = 0,    p_jet = 0; 
    double e_lep    = 0,    e_jet = 0; 

    // angles 
    double phi_mu = 0, theta_mu = 0; 
    double cos_t  = 0, sin_t    = 0; 

    // ---------- Kinematic constants --------- //
    //  w: ((beta_lep/beta_jet) - cos(theta))/sin(theta)
    // o2: w*w + 1 - beta_lep^2 | or | w * w + (m_lep/e_lep)^2
    double w  = 0, o  = 0; // omega, Omega
    double w2 = 0, o2 = 0; // omega^2, Omega^2  
    double wr = 0; // (1 + w*w)^(-0.5)

}; 

class nuclx {
    public: 
        nuclx(particle_template* bjet, particle_template* lep); 
        matrix_t H(double z, double t);
        matrix_t H_tilde(double z, double t); 

        // ---- characteristic polynomial
        double P(double lambda, double t, double z); 
        double dP_dL(double lambda, double t, double z); 

        // ---- lambda values where dP_dL = 0 ---- //
        // !!! warning: this does not imply l0 is an eigenvalue of H_tilde.
        eig_t dPl0(double t, double z); 
        ~nuclx(); 

    private: 
        double G(double t); 

        matrix_t hyperbolic(matrix_t* cx, matrix_t* sx, double t); 
        
        nuclx_t* data = nullptr; 
        particle_template* jet    = nullptr; 
        particle_template* lepton = nullptr; 
}; 


#endif

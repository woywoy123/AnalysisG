#ifndef H_CONUIX_STRUCT
#define H_CONUIX_STRUCT
#include <conuix/solvers.h>
#include <conuix/matrix.h>
class particle_template; 

struct coefficients {
    int  len = 0; 
    long double a = 0; 
    long double b = 0; 
    long double c = 0;
    long double d = 0;
    long double e = 0; 
}; 

struct atomics_t {
    atomics_t(particle_template* jet, particle_template* lep, double m_nu = 0); 

    // ---- Kinematics of the jet and lepton pairs ---- // 
    // === Lepton
    long double beta_l = 0; 
    long double mass_l = 0; 
    long double e_l    = 0; 

    // === jet
    long double beta_j = 0;  
    long double mass_j = 0;
    long double e_j    = 0;  

    // ---- angle between bjet and lepton ---- //
    long double cos = 0; 
    long double sin = 0; 

    // ---- Kinematic variables ---- // 
    long double w = 0;  
    long double o = 0;  

    // ---- rotation matrix ------- //
    long double phi_mu    = 0;
    long double theta_mu  = 0; 
    matrix_t vec_jet; 
    matrix_t R_T; 


    // ---- Reverse Mapping from theta to psi
    long double p_psi_sin = 0;
    long double m_psi_sin = 0;
    long double p_psi_cos = 0;
    long double m_psi_cos = 0; 

    // ---- Z^2 surface polynomial ---- //
    coefficients Z2; 
    
    // ----- Hyperbolic rotation ------ //
    long double cpsi = 0;
    long double spsi = 0;
    long double tpsi = 0;

    // ----- Sx and Sy ----- //
    // Sx(t, Z) = |Z| * (a_x * cosh(t) + b_x * sinh(t)) + c_x;
    // a_x: (o / b_mu) * cos(psi)
    // b_x: - sin(psi)
    // c_x: - m^2_mu / p_mu
    coefficients Sx;


    // Sy(t, Z) = |Z| * (a_y * cosh(t) + b_y * sinh(t)) + c_y;
    // a_y: (o / b_mu) * sin(psi)
    // b_y:  cos(psi)
    // c_y: - w * E^2_mu / p_mu
    coefficients Sy; 

    // ----- Define HBAR -> is effectively H_tilde ----- //
    // H_bar = 1.0/O * [ 
    //      HBX + (beta_mu / sqrt(1 + w^2)) * HBC * cosh(t) + (O / sqrt(1 + w^2)) * HBS * sinh(t) 
    // ];
    //
    // HBX = [ 
    //  [1, 0,  0], 
    //  [w, 0,  0], 
    //  [0, O,  0] 
    // ]
    matrix_t HBX;

    // HBS = [ 
    //  [0, 0, -1], 
    //  [0, 0, -w], 
    //  [0, 0,  0] 
    // ]
    matrix_t HBS; 

    // HBC = [ 
    //  [0, 0, -w], 
    //  [0, 0,  1], 
    //  [0, 0, 0] 
    // ]
    matrix_t HBC;

    // ----- Define HMatrix --------- //
    matrix_t HMX;
    matrix_t HMS;
    matrix_t HMC; 


    // ....... other stuff ............ //
    // gxx: beta_mu * sin(psi) * cosh(tau) + Omega * cos(psi) * sinh(tau)
    long double gxx_a = 0;
    long double gxx_b = 0; 

    // gtx: Omega * cos(psi) * tanh(tau) - beta_mu * sin(psi)
    long double gtx_a = 0;
    long double gtx_b = 0; 

    // characteristic polynomial coefficients of H_tilde
    long double p_a = 0;
    long double p_b = 0;
    long double p_c = 0;
    long double p_d = 0; 

    // derivative of polynomial w.r.t lambda.
    long double dpdl_a = 0; 
    long double dpdl_b = 0;
    long double dpdl_c = 0;

    // derivative of polynomial w.r.t scale.
    long double dpdz_a = 0; 
    long double dpdz_b = 0;
    long double dpdz_c = 0;

    // derivative of polynomial w.r.t tau.
    long double dpdt_a = 0; 
    long double dpdt_b = 0;

    // ------------- Mobius Transformation ------------- //
    // These expressions are derived from the sextic polynomial 
    // that emerges when we solve for tanh(tau) under the condition 
    // dP/dtau = 0 and P = 0.
    // !!!!!! This is an incredibly dense section !!!!!! //
    //  
    //          Omega * cos(psi) - beta_mu * sin(psi) * tanh(tau)
    // M(tau) = -------------------------------------------------
    //          Omega * sin(psi) + beta_mu * cos(psi) * tanh(tau)
    //
    long double M_pm = 0;
    long double M_pp = 0;
    long double M_km = 0;
    long double M_kp = 0;

    coefficients M_qrt;
}; 

#endif



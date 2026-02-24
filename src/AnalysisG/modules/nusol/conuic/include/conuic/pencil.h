#ifndef H_PENCIL_CONUIC
#define H_PENCIL_CONUIC
#include <common/matrix.h>

struct kinematic_c; 
struct angular_t; 

struct Z2_t {
    ~Z2_t(); 
    
    Z2_t(); 
    Z2_t(int s1, int s2, kinematic_c* data); 

    // ...... Pencil function for this branch 
    // Z^2 = a Sx^2 + 2 b Sx Sy + c Sy^2 + d Sx + e
    long double Z2(long double sx, long double sy, long double m_nu = 0); 

    // Z^2 using the hyperbolic parameterization 
    long double Z2(long double tau, long double m_nu = 0); 

    // Mapping to tau given Sx and Sy.
    long double tau(long double sx, long double sy); 

    std::complex<long double> Sx(long double tau, long double m_nu = 0); 
    std::complex<long double> Sy(long double tau, long double m_nu = 0); 

    void print(); 

    // Coefficients
    long double a = 0;
    long double b = 0;
    long double c = 0;
    long double d = 0; 
    long double e = 0; 

    long double O    = 0;
    long double O2   = 0;

    long double e_mu  = 0; 
    long double b_mu  = 0;
    long double m_mu  = 0;
    long double b2_mu = 0; 
    long double m2_mu = 0; 

    long double p_mu  = 0;

    long double Sx0 = 0; 
    long double Sy0 = 0; 
    long double Sz0 = 0; 
   
    branches_t lambda; 

    matrix_t S0; // center of hyperbolic functions
    matrix_t ME; // Eigenvalue Matrix
    matrix_t RK; // Eivenvector Matrix
    matrix_t MK; // Eigenvalue @ Eigenvector

    // ------ H_tilde ------ //
    matrix_t* HTX; // Constants
    matrix_t* HTC; // sosh
    matrix_t* HTS; // sinh

    // ------- H matrix ------- //
    matrix_t* HX; // Constants
    matrix_t* HC; // sosh
    matrix_t* HS; // sinh  

    matrix_t* rot = nullptr; 

    long double eps = 0; // branch sign
    long double kps = 0; // omega branch
    
    angular_t phi; 
};

#endif

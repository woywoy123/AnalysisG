#ifndef MULTISOL_CONUIC_DATA_H
#define MULTISOL_CONUIC_DATA_H

#include <common/matrix.h>
#include <templates/particle_template.h>

struct kinematics_t {
    kinematics_t(particle_template* ptr); 
    ~kinematics_t(); 

    long double px = 0;
    long double py = 0;
    long double pz = 0;     
    long double e  = 0; 
    long double m  = 0;
    long double b  = 0;
    long double p  = 0; 
   
    matrix_t* RT = nullptr;  
    particle_template* ptr_ = nullptr; 
}; 


struct geometry_t {
    // -- we want to remove this angle.
    long double alpha = 0; // skew angle 

    // mutual axis coefficients:
    // a (Sx - Sx0) + b ( Sy - (Sy0[+] + Sy0[-]) / 2 )
    long double l1 = 0; 
    long double l2 = 0;

    // Common center 
    long double Sx0 = 0;
    long double Sy0 = 0;

    // distance between sheets. 
    // mid = d/2
    long double d = 0; 
    long double tau = 0; 
    long double m_nu = 0; 
}; 



struct special_t {
    // cos(phi) tanh(tau) - output of m_nueq_line
    // see constants.h 
    long double nueq_dLpp = 0;  // + sheet 
    long double nueq_dLmm = 0;  // - sheet 

    // swapped 
    long double nueq_dLpm = 0;  // +, - sheet | delta
    long double nueq_dLmp = 0;  // -, + sheet | delta
}; 


struct branches_t {
    branches_t();
    ~branches_t(); 
    long double w = 0; 
    long double O = 0;
    long double bl = 0; 

    // Z2 coefficients 
    long double A = 0; 
    long double B = 0; 
    long double C = 0; 
    long double D = 0; 
    long double E = 0;

    // eigenvalues
    long double l1 = 0; 
    long double l2 = 0; 

    // Sx and Sy solutions
    // --- these are rotations
    long double cpsi = 0;
    long double tpsi = 0;
    long double spsi = 0; 

    // --- center 
    long double sx0 = 0;
    long double sy0 = 0;

    // --- H_tilde matrices
    matrix_t* CC = nullptr;
    matrix_t* SC = nullptr; 
    matrix_t* SS = nullptr; 

    // theta_bmu just in case we need
    long double tth = 0; 
    long double sth = 0; 
    long double cth = 0; 

}; 

struct delta_t {

    // delta roots
    long double dp = 0;
    long double dm = 0;

    // Gammas 
    long double Gp = 0;
    long double Gm = 0; 

    //alpha +
    long double alp  = 0;
    long double salp = 0; 
    long double calp = 0; 
    long double talp = 0; 

    //alpha -
    long double alm  = 0;
    long double salm = 0; 
    long double calm = 0; 
    long double talm = 0; 
    
    // ----- rotation angles ---- //
    long double alpha_p = 0; // -(alm + alp) * 0.5
    long double alpha_m = 0; //  (alp - alm) * 0.5

    // eigenvalues these digonalize 
    // the dG^2 = Gm Gp (Sx - dm Sy) ( Sx - dp Sy)
    // quadric
    long double lp = 0; 
    long double lm = 0; 

    // eigenvalues with Gamma scaling
    // G is a bit misleading it is -Gm x Gp
    long double Glp = 0;
    long double Glm = 0; 
}; 


struct points_t {
    points_t(long double sx, long double sy, long double sz); 
    long double sx = 0;
    long double sy = 0;
    long double sz = 0;
}; 


struct hyper_t {
    hyper_t(long double tau); 

    long double cosh = 0;
    long double sinh = 0;
    long double tanh = 0;

    matrix_t* rx = nullptr;
    matrix_t* ry = nullptr; 
    matrix_t* rz = nullptr; 
}; 

struct angular_t {
    angular_t(long double kappa); 

    long double cos = 0;
    long double sin = 0;
    long double tan = 0;
    matrix_t Rx(); 
    matrix_t Ry(); 
    matrix_t Rz(); 

}; 


#endif 




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
    // the dG^2 = Gm Gp (Sx - dm Sy) ( Sx - dp Sy) quadric
    long double lp = 0; 
    long double lm = 0; 

    // eigenvalues with Gamma scaling
    // G is a bit misleading it is -Gm x Gp
    long double Glp = 0;
    long double Glm = 0; 
}; 


struct cline_t {
    // The delta line from delta G^2.
    // (Sx - dt+- Sy) can be expressed in terms of hyperbolics
    cline_t(branches_t* br, long double dt, double eps, long double dti);
    long double fx(long double m_nu, long double tau, long double phi);
    long double DfxDphi(long double m_nu, long double tau, long double phi); // derivative w.r.t phi.
    long double DfxDtau(long double m_nu, long double tau, long double phi); // derivatve w.r.t tau.
    long double JacoDet(long double m_nu, long double tau, long double phi);
    matrix_t Jacobian(long double m_nu, long double tau, long double phi); 
    long double tau_degenJc(long double phi);  // special case where the eigenvalues become degenerate.
   
    long double center = 0;

    long double alpha   = 0;
    long double beta    = 0;
    long double theta   = 0;

    long double alpha_  = 0;
    long double beta_   = 0;
    long double theta_  = 0; 

    long double tn     = 0; 
    long double cn     = 0; 
    long double sn     = 0; 
    long double r      = 0;

    // special cases:
    // DfDtau and DfDphi = 0, tanh(tau*) = beta / alpha
    long double zero_dd = 0; 

    //the product of the root lines has a zero derivative
    long double delta_lxlm = 0; 

    // Jacobian determinant 
    long double Jdet = 0; 

}; 

struct dline_t {
    // change of basis - now we use the delta lines as the true center.
    // The original Sx and Sy geometry is not properly centered.
    dline_t(kinematics_t* kl, delta_t* dt, branches_t* br, int eps); 

    // output is NOT in terms of Sx or Sy.
    long double dx(long double m_nu, long double tau, long double phi); 
    long double dy(long double m_nu, long double tau, long double phi); 

    // implied Sx or Sy translation layer between dx and dy.
    long double Sdx(long double m_nu, long double tau, long double phi); 
    long double Sdy(long double m_nu, long double tau, long double phi); 

    long double U(long double m_nu, long double tau, long double phi); 
    long double V(long double m_nu, long double tau, long double phi); 


    long double lx0 = 0;
    long double lcx = 0; 
    long double lsx = 0;

    long double ly0 = 0;
    long double lcy = 0; 
    long double lsy = 0;
    
    long double dtp = 0;
    long double dtm = 0; 
    long double lb  = 0; 

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




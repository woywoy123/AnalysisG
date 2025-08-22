#ifndef MULTISOL_H
#define MULTISOL_H

#include <templates/particle_template.h>
#include <reconstruction/matrix.h> 
#include <iostream>
#include <string>

struct multisol_t {
    double A = 0, B = 0, C = 0; 
    double D = 0, E = 0, F = 0; 
   
    // shift of surface 
    double s0x = 0, s0y = 0; 

    // rotation angle
    double psi = 0, cpsi = 0, spsi = 0; 

    // lambdas - eigenvalues
    double lmb1 = -1, lmb2 = 0; 

    // parameterization
    double a_x = 0, b_x = 0, c_x = 0;  // Sx
    double a_y = 0, b_y = 0, c_y = 0;  // Sy
    
    // mass parameterization
    double a_w = 0, b_w = 0;  // mW
    double a_t = 0, b_t = 0, c_t = 0;  // mT

    matrix ht1, ht2, ht3; // H-Tilde
    matrix hx1, hx2, hx3; // H-Lab
}; 

struct rev_t {
    double u = 0;
    double v = 0;
    double u_p = 0; 
    double v_p = 0;

    double t = 0; 
    double z = 0; 
}; 

struct mass_t {
    double mt = 0;
    double mw = 0; 
    double sx = 0; 
    double sy = 0; 
}; 

struct geo_t {
    
    ~geo_t();

    // plane parameters - with ellipse center
    vec3 perp1, perp2, center; 
    double px(double u, double v); 
    double py(double u, double v); 
    double pz(double u, double v); 
    
    // line intersection - ellipse x ellipse: L(s) = r0 + s*d;
    vec3 r0, d, _pts1, _pts2;
    double lx(double s); 
    double ly(double s); 
    double lz(double s); 

    bool valid = false; 
    double _s1 = 0, _s2 = 0; 
    double _p1 = 0, _p2 = 0; 
    double _d1 = 0, _d2 = 0;
    double asym = 0; 
    geo_t* nu1 = nullptr; 
    geo_t* nu2 = nullptr; 
}; 

class multisol
{
    public:
        multisol(particle_template* b, particle_template* l); 
        ~multisol(); 
        void make_rt(); 

        // ------ base matrix ------ //
        matrix    H_tilde(double t, double z); 
        matrix dHdt_tilde(double t, double z = 1);

        matrix    H(double t, double z);  
        matrix dHdt(double t, double z = 1); 
        matrix d2Hdt2(double t, double z = 1); 

        vec3 v          (double t, double z, double phi); // neutrino ellipse
        vec3 dv_dt      (double t, double z, double phi); // neutrino ellipse derivative w.r.t "t"
        vec3 dv_dphi    (double t, double z, double phi); // neutrino ellipse derivative w.r.t "phi"
        vec3 d2v_dt_dphi(double t, double z, double phi); // neutrino ellipse derivative w.r.t "t" and "phi"

        // plane / line intersections 
        geo_t  geometry(double t, double z); 
        geo_t  intersection(multisol* nu, double t1, double z1, double t2, double z2); 
        geo_t* intersection(const vec3& r0, const vec3& d, double t, double z); 

        // ellipse center 
        vec3 center(double t, double z); 
        vec3 normal(double t, double z); 
  
        // use as pre-filter to remove non-physical combinations.
        bool eigenvalues(double t, double z, vec3* real = nullptr, vec3* imag = nullptr); 

        // a special case where the rate of change for eigenvalues is defined as:
        // dP(lambda_c, t*) = 0 and P(lambda_c, t*) = 0.
        double dp_dt(); // returns t.

        mass_t masses(double t, double z);
        rev_t  translate(double sx, double sy);

        void export_ellipse(std::string name, double t, double z);
        void export_ellipse(std::string name, double t, double z, int n_points);

    private: 

        matrix*    rt    = nullptr; 
        multisol_t* para = nullptr; 

        double b_lep  = 0; // beta lepton
        double p_lep  = 0; // momentum lepton        
        double m2_lep = 0; // mass^2 of lepton
        double e2_lep = 0; // energy of lepton

        double b_jet  = 0; // beta bjet
        double p_jet  = 0; // momentum bjet
        double m2_jet = 0; // mass^2 of bjet

        // other params
        double phi_mu   = 0;
        double theta_mu = 0;
        vec3 vx_jet; 

        double o = 0; 
        double w = 0; 
}; 


#endif

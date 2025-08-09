#ifndef NUSOL_H
#define NUSOL_H
#include "particle.h"
#include "structs.h"
#include <string>
#include <iostream>

struct revised_t {
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

    matrix ht1, ht2, ht3; 
    matrix hx1, hx2, hx3; 
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


class nusol_rev
{
    public:
        nusol_rev(const particle& b, const particle& l); 
        ~nusol_rev(); 
        void make_rt(); 

        matrix  H(double t,  double z) const; 
        matrix  h_tilde(double t,  double z); 

        geo_t  geometry(double t, double z); 
        geo_t  intersection(const nusol_rev* nu, double t1, double z1, double t2, double z2) const; 
        geo_t* intersection(const vec3& r0, const vec3& d, double t, double z) const; 

        vec3 center(double t, double z) const; 
        vec3 normal(double t, double z) const; 

        mass_t masses(double t, double z);
        rev_t  translate(double sx, double sy);

        vec3 v(double t, double z, double phi); 
        void export_ellipse(std::string name, double t, double z);
        void export_ellipse(std::string name, double t, double z, int n_points);

    private: 

        particle*  bjet = nullptr; 
        particle*  lep  = nullptr; 
        matrix*    rt   = nullptr; 
        revised_t* para = nullptr; 
        double o = 0; 
        double w = 0; 
}; 


#endif

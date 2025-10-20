#include <conics/nusol.h>
#include <complex>

nuclx::nuclx(particle_template* bjet, particle_template* lep){
    this -> jet = bjet; this -> lepton = lep; 
    this -> data = new nuclx_t(bjet, lep); 

}

nuclx::~nuclx(){
    delete this -> data; 
}

matrix_t nuclx::hyperbolic(matrix_t* cx, matrix_t* sx, double t){
    return *cx * std::cosh(t) + *sx * std::sinh(t); 
}

matrix_t nuclx::H(double z, double t){
    return z * (this -> data -> Hc + this -> hyperbolic(&this -> data -> H1, &this -> data -> H2, t)); 
}

matrix_t nuclx::H_tilde(double z, double t){
    return z * (this -> data -> HBc + this -> hyperbolic(&this -> data -> HB1, &this -> data -> HB2, t)); 
}

double nuclx::P(double lbd, double t, double z){
    double l1 = lbd; double l2 = l1*l1; double l3 = l2 * l1; 
    double z1 = z;   double z2 = z1*z1; double z3 = z2 * z1; 

    double sh = std::sinh(t); double ch = std::cosh(t); 
    double cx = l3 + this -> data -> a_l * l2 * z1; 
    cx += (this -> data -> b_l * ch + this -> data -> c_l * sh) * z2 * l1; 
    return cx + this -> data -> d_l * sh * z3; 
}

double nuclx::dP_dL(double lbd, double t, double z){
    double l1 = lbd; double l2 = l1*l1; 
    double z1 = z;   double z2 = z1*z1; 

    double sh = std::sinh(t); double ch = std::cosh(t); 
    return 3 * l2 + 2 * this -> data -> a_l * l1 * z1 + (this -> data -> b_l * ch + this -> data -> c_l * sh) * z2; 
}

eig_t nuclx::dPl0(double t, double z){
    std::complex<double> a = z / (3 * this -> data -> o);
    std::complex<double> b = 1 - 3 * (this -> data -> o / (1 + this -> data -> w2)) * this -> G(t); 
    std::complex<double> l1 = std::pow(b, 0.5); 
    l1 = a * (1 + l1); l2 = a * (1 - l1);

    eig_t n; 
    n.r1 = l1.real(); n.i1 = l1.imag(); 
    n.r2 = l2.real(); n.i2 = l2.imag();
    return n; 
}

double nuclx::G(double t){
    double a = this -> data -> beta_lep * this -> data -> w * std::cosh(t);
    double b = this -> data -> o * std::sinh(t); 
    return a - b; 
}

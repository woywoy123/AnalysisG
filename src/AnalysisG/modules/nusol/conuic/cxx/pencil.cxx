#include <conuic/variables.h>
#include <conuic/angular.h>
#include <conuic/atomics.h>
#include <conuic/pencil.h>
#include <math.h>

matrix_t* Make_H(int type, bool tilde, Z2_t* data){
    matrix_t A = matrix_t(3, 3);
    if (type == 0){
        A.at(0, 0) = 1.0;
        A.at(1, 0) = data -> phi.tan; 
        A.at(2, 1) = data -> O; 
    }

    if (type == 1){
        A.at(0, 2) = - data -> b_mu * data -> phi.cos; 
        A.at(1, 2) = - data -> b_mu * data -> phi.sin; 
    }

    if (type == 2){
        A.at(0, 2) = - data -> O * data -> phi.sin; 
        A.at(1, 2) = - data -> O * data -> phi.cos; 
    }
    if (!tilde){return new matrix_t(data -> rot -> dot(A));}
    return new matrix_t(A); 
}

Z2_t::~Z2_t(){
    flush(&this -> HTX); 
    flush(&this -> HTC);
    flush(&this -> HTS); 

    flush(&this -> HX);
    flush(&this -> HC);
    flush(&this -> HS); 
}

Z2_t::Z2_t(){}

Z2_t::Z2_t(int s1, int s2, kinematic_c* data){
    this -> rot = data -> rot; 
    this -> eps = s2;  // <..... branch sign
    this -> kps = s1; 

    // ...... Constants ....... //
    this -> phi  = angular_t(data -> w.pair(this -> kps), false, false, true); 
    this -> O    = data -> O.pair(this -> kps); 
    this -> b_mu = data -> b_mu; 
    this -> p_mu = data -> p_mu; 
    this -> m_mu = data -> m_mu; 
    this -> e_mu = data -> e_mu; 

    this -> b2_mu = this -> b_mu * this -> b_mu;
    this -> m2_mu = this -> m_mu * this -> m_mu; 

    this -> O2    = this -> O * this -> O;

    // ......... Z^2 polynomial coefficients ........ //
    this -> a = ( this -> b2_mu - std::pow(this -> phi.tan, 2) ) / this -> O2;
    this -> b =     2 * this -> phi.tan / this -> O2; 
    this -> c = - ( 1 - this -> b2_mu ) / this -> O2; 
    this -> d = 2 * this -> p_mu;
    this -> e = this -> m2_mu;

    // .......... Sx and Sy hyperbolic parameterization ......... //
    // Sx0 and Sy0 are the center of the surfaces.
    this -> Sx0 = -  this -> m2_mu / this -> p_mu; 
    this -> Sy0 = - (this -> e_mu  / this -> b_mu)* this -> phi.tan; 

    this -> S0 = matrix_t(2, 1);
    this -> S0.at(0,0) = this -> Sx0;
    this -> S0.at(1,0) = this -> Sy0;

    // ........... Sx and Sy in matrix form ................. //
    this -> ME = matrix_t(2, 2);
    this -> ME.at(0, 0) =  this -> O / this -> b_mu;   // <.... Eigenvalues
    this -> ME.at(1, 1) = -1;                          // <.... Eigenvalues
    this -> lambda = branches_t(this -> O2 / this -> b2_mu, -1, "lambda"); 


    this -> RK = matrix_t(2, 2); 
    this -> RK.at(0, 0) =  this -> phi.cos * this -> eps; // <.... s2 is the 
    this -> RK.at(1, 0) =  this -> phi.sin * this -> eps;
    this -> RK.at(0, 1) = -this -> phi.sin;
    this -> RK.at(1, 1) =  this -> phi.cos;

    this -> MK  =  this -> RK.dot(this -> ME);

    // This generates the H_tilde and H matrices.  
    this -> HTX = Make_H(0, true, this); // constant matrix
    this -> HTC = Make_H(1, true, this); // Cosh matrix
    this -> HTS = Make_H(2, true, this); // Sinh matrix

    this -> HX = Make_H(0, false, this);
    this -> HC = Make_H(1, false, this); 
    this -> HS = Make_H(2, false, this); 
}

std::complex<long double> Z2_t::Sx(long double tau, long double m_nu){
    std::complex<long double> _a = (2 * m_nu * m_nu - this -> Sz0) * std::pow(this -> O/this -> b_mu, 2); 
    std::complex<long double> _b = (2 * m_nu * m_nu - this -> Sz0); 
    _a = std::sqrt(_a); _b = std::sqrt(_b); 

    hyper_t hx(tau); 
    return this -> phi.cos * (this -> eps * _a * hx.cosh - this -> phi.tan * _b * hx.sinh) + this -> Sx0; 
}

std::complex<long double> Z2_t::Sy(long double tau, long double m_nu){
    std::complex<long double> _a = (2 * m_nu * m_nu - this -> Sz0) * std::pow(this -> O/this -> b_mu, 2); 
    std::complex<long double> _b = (2 * m_nu * m_nu - this -> Sz0); 
    _a = std::sqrt(_a); _b = std::sqrt(_b); 

    hyper_t hx(tau); 
    return this -> phi.cos * (this -> eps * this -> phi.tan * _a * hx.cosh + _b * hx.sinh) + this -> Sy0; 
}

long double Z2_t::Z2(long double tau, long double m_nu){
    if (m_nu < 0){m_nu = std::abs(this -> Sz0) + 1;}

    std::complex<long double> sx = this -> Sx(tau, m_nu); 
    std::complex<long double> sy = this -> Sy(tau, m_nu); 
    return this -> Z2(sx.real(), sy.real(), m_nu);    
}

long double Z2_t::Z2(long double sx, long double sy, long double m_nu){
    long double Z = this -> a * sx * sx; 
    Z += this -> b * sx * sy;
    Z += this -> c * sy * sy;
    Z += this -> d * sx;
    Z += this -> e - m_nu * m_nu; 
    return Z; 
}

long double Z2_t::tau(long double sx, long double sy){
    matrix_t S = shift_t(sx, sy).to_mat(); 
    S = (this -> MK.inv()).dot(S - this -> S0); 
    long double t = S.at(1,0) / S.at(0,0); 
    return std::atanh((std::abs(t) > 1) ? 1.0 / t : t); 
}

void Z2_t::print(){
    std::cout << std::endl;
    debug_s("A", this -> a, this -> kps > 0); 
    debug_s("B", this -> b, this -> kps > 0);
    debug_s("C", this -> c, this -> kps > 0);
    debug_s("D", this -> d, this -> kps > 0);
    debug_s("E", this -> e, this -> kps > 0); 
    std::cout << std::endl;
}



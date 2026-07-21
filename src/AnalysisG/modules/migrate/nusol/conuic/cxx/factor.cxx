#include <conuic/variables.h>
#include <conuic/constants.h>
#include <conuic/factor.h>
#include <math.h>

G2_t::G2_t(){}

G2_t::G2_t(kinematic_c* data){
    this -> b_mu = data -> b_mu;
    this -> p_mu = data -> p_mu; 
    this -> m_mu = data -> m_mu; 
    this -> b2_mu = this -> b_mu * this -> b_mu;
    this -> m2_mu = this -> m_mu * this -> m_mu; 

    // ---------------- Constants ----------------------- //
    this -> w  = branches_t(omega(+1, data), omega(-1, data), "omega"); 
    this -> O  = branches_t(Omega(+1, data), Omega(-1, data), "Omega");  

    // ---------------- Factorization constants --------- //
    this -> G     = branches_t(Gamma(+1, data), Gamma(-1, data), "Gamma"); // factor
    this -> delta = branches_t(dG2_delta(+1, data), dG2_delta(-1, data), "delta"); // roots
    this -> kappa = branches_t(
            angular_t(this -> delta.p, false, false, true), 
            angular_t(this -> delta.m, false, false, true), 
            "kappa"
    ); 
    this -> psi   = branches_t(
            angular_t((this -> kappa.pA.phi + this -> kappa.mA.phi) * 0.5), 
            angular_t((this -> kappa.pA.phi - this -> kappa.mA.phi) * 0.5),
            "delta psi"
    ); 

    this -> lambda = branches_t(dG2_lambda(+1, this), dG2_lambda(-1, this), "lambda"); 

    this -> eigM = matrix_t(2, 2); 
    this -> eigM.at(0,0) =  1;
    this -> eigM.at(0,1) = -(this -> psi.pA.tan + this -> psi.mA.tan) * 0.5;
    this -> eigM.at(1,0) = -(this -> psi.pA.tan + this -> psi.mA.tan) * 0.5;
    this -> eigM.at(1,1) =   this -> psi.pA.tan * this -> psi.mA.tan;
    this -> MK = - this -> G.m * this -> G.p * this -> eigM; 
}

shift_t G2_t::dG2(long double tau){
    hyper_t h = hyper_t(tau); 
    matrix_t S = this -> MK.dot(h.mat); 
    return shift_t(S.at(0,0), S.at(1, 0), this -> dG2(S.at(0,0), S.at(1,0))); 
}

long double G2_t::tau(long double sx, long double sy){
    matrix_t s = shift_t(sx, sy).to_mat(); 
    matrix_t S = (this -> MK.inv()).dot(S); 
    long double t = S.at(1,0) / S.at(0,0); 
    return std::atanh((std::abs(t) > 1) ? 1.0 / t : t); 
}

long double G2_t::dG2(long double sx, long double sy){
    return - this -> G.m * this -> G.p * (sx - this -> delta.p * sy)*(sx - this -> delta.m * sy);
}


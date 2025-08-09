#include "nusol_ref.h"
#include <iostream>
#include <fstream>
#include <math.h>

nusol::nusol(const particle& b, const particle& l){
    this -> bjet = new particle(b); 
    this -> lep  = new particle(l); 
}

reference_t nusol::update(double mT_, double mW_){
    this -> mW2 = mW_ * mW_;
    this -> mT2 = mT_ * mT_; 

    double c = cos_theta(this -> bjet, this -> lep); 
    double s = std::pow(1.0 - c*c, 0.5);
    double w  = (this -> lep -> b / this -> bjet -> b - c)/s;
    double o2 = w*w + 1 - this -> lep -> b2; 
    double o  = std::pow(o2, 0.5); 
    double x0  = -(this -> mW2 - this -> lep -> m2) / (2.0 * this -> lep -> e); 
    double sx  = (x0 * this -> lep -> b - this -> lep -> p * (1 - this -> lep -> b2)) / this -> lep -> b2; 

    double x0p = -(this -> mT2 - this -> mW2 - this -> bjet -> m2) / (2.0 * this -> bjet -> e); 
    double sy  = (x0p / this -> bjet -> b - c * sx) / s; 
    double x1  = sx - (sx + w * sy) / o2; 
    double y1  = sy - (sx + w * sy) * w / o2; 
    double t1_ =  std::pow(x1, 2) * o2;
    double t2_ =  std::pow(sy - w * sx, 2);
    double t3_ = -std::pow(x0, 2) + this -> lep -> b2 * this -> mW2; 
    double z2  = t1_ - t2_ - t3_;
    double z   = std::pow(z2, 0.5); 
    this -> make_rt();

    matrix HT(3, 3);
    HT.at(0, 0) = z/o; 
    HT.at(1, 0) = w * z/o; 
    HT.at(2, 1) = z; 
    HT.at(0, 2) = x1 - this -> lep -> p; 
    HT.at(1, 2) = y1;
    matrix H = (*this -> rt) * HT;
    
    reference_t out;
    out.x0p = x0p; 
    out.x0  = x0;
    out.x1  = x1;
    out.y1  = y1; 
    out.Sx  = sx; 
    out.Sy  = sy;
    out.Z2  = z2;  
    out.Z   = z ;
    out.HT  = HT; 
    out.H   = H; 
    return out; 
}

nusol::~nusol(){
    safe(&this -> bjet);
    safe(&this -> lep);
    safe(&this -> rt); 
}

void nusol::make_rt(){
    if (this -> rt){return;}
    double phi_mu   = std::atan2(this -> lep -> py, this -> lep -> px);
    double theta_mu = std::acos( this -> lep -> pz / this -> lep -> p);
    matrix Rz(3,3); 
    Rz.at(0,0) =  std::cos(phi_mu); 
    Rz.at(0,1) = -std::sin(phi_mu); 
    Rz.at(2,2) = 1;
    Rz.at(1,0) = std::sin(phi_mu);
    Rz.at(1,1) = std::cos(phi_mu);
    
    matrix Ry(3,3); 
    Ry.at(0,0) = std::cos(theta_mu); 
    Ry.at(0,2) = std::sin(theta_mu); 
    Ry.at(1,1) = 1;
    Ry.at(2,0) = -std::sin(theta_mu); 
    Ry.at(2,2) =  std::cos(theta_mu);

    vec3 b_p = Ry * (Rz * vec3{this -> bjet -> px, this -> bjet -> py, this -> bjet -> pz});
    double alpha = -std::atan2(b_p.z, b_p.y);

    matrix Rx(3,3);
    Rx.at(0,0) = 1; 
    Rx.at(1,1) =  std::cos(alpha); 
    Rx.at(1,2) = -std::sin(alpha);
    Rx.at(2,1) =  std::sin(alpha); 
    Rx.at(2,2) =  std::cos(alpha);
    this -> rt = new matrix(Rz.T() * Ry.T() * Rx.T()); 
}

vec3 reference_t::nu(double phi){return this -> H * vec3{std::cos(phi), std::sin(phi), 1};}

#include <conuic/variables.h>

kinematic_c::~kinematic_c(){
    flush(&this -> rot); 
}

kinematic_c::kinematic_c(
    particle_template* jet_, particle_template* lep_
){
    this -> jet = jet_; this -> lep = lep_; 
    this -> theta = angular_t(std::acos(angles(jet_, lep_, this))); 
    
    this -> w  = branches_t(omega(+1, this), omega(-1, this), "w");
    this -> O  = branches_t(Omega(+1, this), Omega(-1, this), "O");

    this -> Zpp = Z2_t(+1, +1, this);
    this -> Zpm = Z2_t(+1, -1, this);
    this -> Zmp = Z2_t(-1, +1, this);
    this -> Zmm = Z2_t(-1, -1, this);
    this -> G2  = G2_t(this); 

    this -> debug(); 
}


void kinematic_c::debug(){
    std::string hx = std::string(this -> jet -> hash); 
    if (hx != "0x791626d00b92e24c"){return;}
    hx += "- " + std::string(this -> lep -> hash); 
    long double sx = -25379.016037398775;
    long double sy = -28841.223092821998; 

    long double nu = 16.981686095870792; 
    long double wb = 79530.04248810474; 
    long double tp = 171832.96610250633; 

    long double tau = this -> Zpp.tau(sx, sy); 
    this -> Zpp.print(); 
    this -> Zmm.print(); 
    debug_s("Z2pp", this -> Zpp.Z2(sx, sy, nu), true);
    debug_s("Z2pm", this -> Zpm.Z2(sx, sy, nu), true);
    debug_s("Z2mp", this -> Zmp.Z2(sx, sy, nu), true);
    debug_s("Z2mm", this -> Zmm.Z2(sx, sy, nu), true);
    std::cout << std::endl; 

    debug_s(this -> w); 
    debug_s(this -> O); 
    debug_s(this -> G2.delta); 
    debug_s(this -> G2.G);
    debug_s(this -> G2.lambda); 

    debug_s("dZ"  , this -> G2.dG2(sx, sy), (this -> Zpp.Z2(sx, sy, nu) - this -> Zmm.Z2(sx, sy, nu)), 0.0001); 
    debug_s("root", this -> G2.delta.p * this -> G2.delta.m, -(1 - std::pow((long double)this -> lep -> beta, 2)), 0.00001); 

    abort(); 
}


#include <conuic/conuic.h>
#include <conuic/angular.h>
#include <conuic/atomics.h>

conuic::conuic(particle_template* jet, particle_template* lepton){
    this -> lep_ = new kinematics_t(lepton); 
    this -> jet_ = new kinematics_t(jet); 
    this -> RT_  = angles(jet, lepton); 
    this -> shr   = new shared_t(this -> jet_, this -> lep_);

    this -> plus  = new base_t(this -> shr, +1); 
    this -> minus = new base_t(this -> shr, -1); 
    this -> pl1   = new pk1l_t(this -> plus, this -> minus); 

}

conuic::~conuic(){
    flush(&this -> lep_);
    flush(&this -> jet_);   
    flush(&this -> RT_); 

    flush(&this -> shr); 
    flush(&this -> plus);   
    flush(&this -> minus); 
    flush(&this -> pl1);
}

base_t* conuic::branching(int s){
    return (s < 0) ? this -> minus : this -> plus; 
}

void conuic::proof(particle_template* nu){
    debug_t dbs = debug_t(); 
    particle_template wbs = *nu + *this -> lep_ -> ptr_; 
    particle_template top = wbs + *this -> jet_ -> ptr_; 

    long double tmass = top.mass;
    long double wmass = wbs.mass; 
    long double nu_ms = nu -> mass; 
    
    long double sx = this -> Sx(wmass, nu_ms);
    long double sy = this -> Sy(tmass, wmass, nu_ms); 

    long double wm_ = this -> mW2(sx, nu_ms);
    long double tm_ = this -> mT2(sx, sy, nu_ms); 
    dbs.assertions("mW2", wmass * wmass, wm_); 
    dbs.assertions("mT2", tmass * tmass, tm_); 

    long double z2p = this -> Z2(sx, sy, nu_ms, +1); 
    long double z2m = this -> Z2(sx, sy, nu_ms, -1); 
    //dbs.assertions("dg2", z2p - z2m, this -> g2(sx, sy)); 
    //dbs.assertions("dG2", z2p - z2m, this -> G2(sx, sy)); 
   
    shared_t* sh = this -> shr; 
    pk1l_t*  p1l = this -> pl1; 
    //dbs.assertions("d[+]d[-]", pw(sh -> b_mu), 1.0L + p1l -> dp * p1l -> dm); 
    //dbs.assertions("d[-] =  sqrt(1 - b^2_mu) tanh(eta)",  p1l -> dm,  std::sqrt(1 - pw(sh -> b_mu)) * std::tanh(p1l -> eta)); 
    //dbs.assertions("d[+] = -sqrt(1 - b^2_mu) coth(eta)",  p1l -> dp, -std::sqrt(1 - pw(sh -> b_mu)) / std::tanh(p1l -> eta)); 

    //dbs.assertions("Sx -> Lx -> Sx", sx, p1l -> Sx( p1l -> Lx(sx, sy), p1l -> Ly(sx, sy) )); 
    //dbs.assertions("Sy -> Ly -> Sy", sy, p1l -> Sy( p1l -> Lx(sx, sy), p1l -> Ly(sx, sy) )); 

    //dbs.assertions("Sx -> lx -> Sx", sx, p1l -> sx( p1l -> lx(sx, sy), p1l -> ly(sx, sy) )); 
    //dbs.assertions("Sy -> ly -> Sy", sy, p1l -> sy( p1l -> lx(sx, sy), p1l -> ly(sx, sy) )); 

    long double z2plxy = this -> Z2lxly(p1l -> Lx(sx, sy), p1l -> Ly(sx, sy), nu_ms, +1);
    long double z2mlxy = this -> Z2lxly(p1l -> Lx(sx, sy), p1l -> Ly(sx, sy), nu_ms, -1);

    //dbs.assertions("Z2(lx, ly)[+]", z2p, z2plxy); 
    //dbs.assertions("Z2(lx, ly)[-]", z2m, z2mlxy); 
    //dbs.assertions("Z2(lx, ly)[+] - Z2(lx, ly)[-]", z2p - z2m, z2plxy - z2mlxy); 

    long double z2plpp = this -> Z2lxly(p1l -> L0pp, p1l -> L0pm, nu_ms, +1);
    long double z2mlpp = this -> Z2lxly(p1l -> L0mp, p1l -> L0mm, nu_ms, -1);
    //dbs.assertions("Z2(lx0[+], ly0[+])[+]", -nu_ms * nu_ms, z2plpp); 
    //dbs.assertions("Z2(lx0[-], ly0[-])[-]", -nu_ms * nu_ms, z2mlpp); 

    abort(); 

}; 


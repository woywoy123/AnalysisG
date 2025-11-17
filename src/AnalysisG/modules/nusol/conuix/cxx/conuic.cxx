#include <templates/particle_template.h>
#include <conuix/conuic.h>

conuic::conuic(particle_template* jet, particle_template* lep){
    this -> _jet = jet;
    this -> _lep = lep; 
    this -> cache = new atomics_t(this -> _jet, this -> _lep); 
    this -> tstar = this -> cache -> Mobius.tstar; 
    this -> error = this -> cache -> Mobius.error; 
    this -> converged = this -> cache -> Mobius.converged;

    this -> vstar = matrix_t(3, 1); 
    this -> cache -> eigenvector(this -> cache -> Mobius.tstar, &this -> vstar, &this -> theta);     
}

conuic::~conuic(){
    this -> _jet = nullptr; 
    this -> _lep = nullptr; 
    if (!this -> cache){return;}
    delete this -> cache; 
    this -> cache = nullptr; 
}

void conuic::debug(){
    std::string hx = std::string(this -> _jet -> hash) + "-" + std::string(this -> _lep -> hash);
    if (hx != "0x5a4ac8ead1f7d952-0x1dbf9ed01fdef73b"){return;}
    this -> cache -> debug_mode(this -> _jet, this -> _lep); 
    abort(); 
}

long double conuic::Z2(long double sx, long double sy){
    return this -> cache -> pencil.Z2(sx, sy); 
}

long double conuic::Sx(long double t, long double Z){
    return this -> cache -> Sx.Sx(t, Z); 
}

long double conuic::Sy(long double t, long double Z){
    return this -> cache -> Sy.Sy(t, Z); 
}

long double conuic::x1(long double t, long double Z){
    return this -> cache -> x1(Z, t); 
}

long double conuic::y1(long double t, long double Z){
    return this -> cache -> y1(Z, t); 
}

long double conuic::P(long double l, long double t, long double Z){
    return this -> cache -> Mobius.P(Z, l, t); 
}

long double conuic::dPdt(long double l, long double t, long double Z){
    return this -> cache -> Mobius.dPdt(Z, l, t); 
}

long double conuic::dPdtL0(long double t, long double Z){
    return this -> cache -> Mobius.dPdtL0(Z, t); 
}

long double conuic::dPl0(long double t){
    return this -> cache -> Mobius.dPl0(t, true); 
}

matrix_t conuic::Hmatrix(long double t, long double Z){
    return this -> cache -> H_Matrix.H_Matrix(t, Z); 
}

bool conuic::get_TauZ(long double sx, long double sy, long double* z, long double* t){
    return this -> cache -> GetTauZ(sx, sy, z, t); 
}

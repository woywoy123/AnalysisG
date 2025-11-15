#include <templates/particle_template.h>
#include <conuix/conuic.h>

void conuic::get_tau(long double* v, conuic* c){*v = c -> tau;}
void conuic::set_tau(long double* v, conuic* c){
    c -> tau  = *v; 
    c -> stau = std::sinh(c -> tau); 
    c -> ctau = std::cosh(c -> tau); 
    c -> ttau = std::tanh(c -> tau); 
}

void conuic::set_scl(long double* v, conuic* c){c -> scale = *v;}
void conuic::get_scl(long double* v, conuic* c){*v = c -> scale;}

void conuic::set_lmb(long double* v, conuic* c){c -> lamb = *v;}
void conuic::get_lmb(long double* v, conuic* c){*v = c -> lamb;}


conuic::conuic(particle_template* jet, particle_template* lep){
    this -> _jet = jet;
    this -> _lep = lep; 
    this -> cache = new atomics_t(this -> _jet, this -> _lep); 

    this -> t.set_setter(this -> set_tau); 
    this -> t.set_getter(this -> get_tau); 
    this -> t.set_object(this); 

    this -> z.set_setter(this -> set_scl); 
    this -> z.set_getter(this -> get_scl); 
    this -> z.set_object(this); 

    this -> l.set_setter(this -> set_lmb); 
    this -> l.set_getter(this -> get_lmb); 
    this -> l.set_object(this); 
}

conuic::~conuic(){
    this -> _jet = nullptr; 
    this -> _lep = nullptr; 
    delete this -> cache; 
    this -> cache = nullptr; 
}

void conuic::solve(){
    this -> debug();   
}

void conuic::debug(){
    std::string hx = std::string(this -> _jet -> hash) + "-" + std::string(this -> _lep -> hash);
    if (hx != "0x5a4ac8ead1f7d952-0x1dbf9ed01fdef73b"){return;}
    this -> cache -> debug_mode(this -> _jet, this -> _lep); 
    abort(); 

}


matrix_t conuic::get_RT(){return this -> cache -> rotation.R_T;}

long double conuic::Z2(long double sx, long double sy){
    return this -> cache -> pencil.Z2(sx, sy); 
}

long double conuic::Sx(long double Tau, long double Z){
    return this -> cache -> Sx.Sx(Tau, Z); 
}

long double conuic::Sy(long double Tau, long double Z){
    return this -> cache -> Sy.Sy(Tau, Z); 
}











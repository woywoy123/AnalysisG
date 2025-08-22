#include <conics/nusol.h>

nuclx::nuclx(particle_template* bjet, particle_template* lep){
    this -> jet = bjet; this -> lepton = lep; 
    this -> data = new nuclx_t(bjet, lep); 
}

nuclx::~nuclx(){
    delete this -> data; 
}


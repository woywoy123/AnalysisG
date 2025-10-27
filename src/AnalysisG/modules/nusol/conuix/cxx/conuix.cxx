#include <conuix/conuix.h>
#include <conuix/conuic.h>

conuix::conuix(nusol_t* parameters){
    this -> params = parameters; 
    this -> prefix = "Conuix"; 
    this -> params -> met_x = std::cos(this -> params -> phi) * this -> params -> met;  
    this -> params -> met_y = std::sin(this -> params -> phi) * this -> params -> met;
    size_t n = this -> params -> targets -> size();

    std::vector<particle_template*> jets = {}; 
    std::vector<particle_template*> leps = {};
    jets.reserve(n); 
    leps.reserve(n);

    for (size_t x(0); x < this -> params -> targets -> size(); ++x){
        particle_template* pn = this -> params -> targets -> at(x); 
        if (pn -> is_lep){leps.push_back(pn);}
        else {jets.push_back(pn);}
    }

    int idx = -1; 
    this -> cnx = new std::vector<conuic*>(jets.size() * leps.size(), nullptr);
    for (size_t i(0); i < jets.size(); ++i){
        for (size_t j(0); j < leps.size(); ++j){(*this -> cnx)[++idx] = new conuic(jets[i], leps[j]);}
    }
}

conuix::~conuix(){
    for (size_t x(0); x < this -> cnx -> size(); ++x){
        delete this -> cnx -> at(x); 
        (*this -> cnx)[x] = nullptr; 
    }
    delete this -> cnx; 
    this -> cnx = nullptr; 
}

void conuix::solve(){
    for (size_t x(0); x < this -> cnx -> size(); ++x){
        this -> cnx -> at(x) -> solve(); 
    }
    abort(); 
}


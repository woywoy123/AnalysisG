#include <templates/particle_template.h>
#include <conuic/variables.h>
#include <conuic/conuic.h>

conuiac::conuiac(nusol_t* param){
    this -> params = param; 
    size_t n = this -> params -> targets -> size();
    std::vector<particle_template*> jets = {}; 
    std::vector<particle_template*> leps = {};
    jets.reserve(n); leps.reserve(n);
    for (size_t x(0); x < this -> params -> targets -> size(); ++x){
        particle_template* pn = this -> params -> targets -> at(x); 
        if (pn -> is_lep){leps.push_back(pn);}
        else {jets.push_back(pn);}
    }

    this -> kins.reserve(jets.size() + leps.size()); 
    for (size_t x(0); x < jets.size(); ++x){
        for (size_t y(0); y < leps.size(); ++y){
            this -> kins.push_back(new kinematic_c(jets[x], leps[y])); 
        }
    }
    std::cout << "....." << std::endl;
}

conuiac::~conuiac(){

}

std::vector<particle_template*> conuiac::solve(){





    return {}; 
}



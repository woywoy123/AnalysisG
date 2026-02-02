#include <bsm_4tops/event.h>
#include "multisol/multisol.h"
#include "conuix.h"

conuix::conuix(){
    this -> name = "conuix";
}

conuix::~conuix(){}

selection_template* conuix::clone(){
    conuix* cl = new conuix(); 
    return cl;
}

void conuix::merge(selection_template* sl){
    conuix* slt = (conuix*)sl; 
}

bool conuix::selection(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 
    return true; 
}
   
bool conuix::strategy(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 
    std::vector<particle_template*> detectors = evn -> DetectorObjects;
    std::vector<particle_template*> topchildren = evn -> Children; 

    std::map<int, std::vector<particle_template*>> top_ptr; 
    std::map<int, std::vector<particle_template*>> bos_ptr; 

    for (size_t y(0); y < detectors.size(); ++y){
        particle_template* jx = detectors[y]; 
        std::string type = jx -> type; 
        std::vector<int> topv = {};
        if (type == "jet"){topv = ((jet*)jx) -> top_index;}
        if (type == "mu"){ topv = {((muon*)jx) -> top_index};}
        if (type == "el"){ topv = {((electron*)jx) -> top_index};}
        for (size_t k(0); k < topv.size(); ++k){
            top_ptr[topv[k]].push_back(jx);
            if (type != "mu" && type != "el"){continue;}
            bos_ptr[topv[k]].push_back(jx);
        }
    }

    std::map<int, particle_template*> leptons; 
    std::map<int, particle_template*> neutrinos; 
    std::map<int, std::vector<particle_template*>> jets; 
    for (size_t x(0); x < topchildren.size(); ++x){
        particle_template* tc = topchildren[x]; 
        if (!tc -> is_nu){continue;}
        neutrinos[tc -> index] = tc;
    }

    std::cout << "=============================" << std::endl;
    std::map<int, particle_template*>::iterator itr; 
    for (itr = neutrinos.begin(); itr != neutrinos.end(); ++itr){
        if (!neutrinos[itr -> first]){continue;}
        std::vector<particle_template*> det = top_ptr[itr -> first];
        if (top_ptr[itr -> first].size() < 2){continue;}
                
        particle_template* wpx = nullptr; 
        bos_ptr[itr -> first].push_back(neutrinos[itr -> first]); 
        this -> sum(&bos_ptr[itr -> first], &wpx); 

        particle_template* topx = nullptr; 
        top_ptr[itr -> first].push_back(neutrinos[itr -> first]); 
        this -> sum(&top_ptr[itr -> first], &topx); 

        std::cout << double(topx -> mass) * 0.001 << std::endl; 
        particle_template* nu = itr -> second; 
        std::cout << double(nu -> px) * 0.001 << " "; 
        std::cout << double(nu -> py) * 0.001 << " "; 
        std::cout << double(nu -> pz) * 0.001 << " "; 
        std::cout << double(nu -> e ) * 0.001 << std::endl; 
        std::cout << "_____________________________" << std::endl;
    }
    
    std::cout << "=============================" << std::endl;

    params_t prm; 
    prm.targets = &detectors;
    prm.met = evn -> met; 
    prm.phi = evn -> phi; 
//    prm.mass_t = topx -> mass; 
//    prm.mass_w = wpx -> mass; 
    multisol mt = multisol(&prm); 
    mt.prescan(); 
    std::cout << "...." << std::endl;

    abort(); 
    return true; 
}



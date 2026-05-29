#include <bsm_4tops/event.h>
#include <conuic/conuic.h>
#include "multisol.h"

multisol::~multisol(){}
multisol::multisol(){this -> name = "multisol";}
selection_template* multisol::clone(){return new multisol();}

void multisol::merge(selection_template* sl){
    multisol* slt = (multisol*)sl; 
}

bool multisol::selection(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 
    return true; 
}
   
bool multisol::strategy(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 
    std::vector<particle_template*> detectors = evn -> DetectorObjects;
    std::vector<particle_template*> topchildren = evn -> Children; 

    std::map<int, particle_template*> bs_; 
    std::map<int, particle_template*> ls_; 
    std::map<int, particle_template*> nus_; 
    for (int i(0); i < topchildren.size(); ++i){
        particle_template* cl = topchildren[i]; 
        int tdx = cl -> as<top_children>() -> top_index; 
        if (cl -> is_nu){nus_[tdx] = cl;}
        else if (cl -> is_lep){ls_[tdx] = cl;}
        else {bs_[tdx] = cl;}
    }
    
    std::cout << "------------- EVENT ----------------" << std::endl; 
    std::map<int, particle_template*>::iterator itn; 
    for (itn = nus_.begin(); itn != nus_.end(); ++itn){
        particle_template* pn = itn -> second; 
        particle_template* ln = ls_[itn -> first]; 
        particle_template* bn = bs_[itn -> first]; 
        std::cout << "==========================" << std::endl;
        std::cout << "---------- neutrino ---------- " << std::endl; 
        std::cout << " px: "  << tools::to_string(pn -> px, 12) 
                  << " py: " << tools::to_string(pn -> py, 12)
                  << " pz: " << tools::to_string(pn  -> pz, 12) 
                  << " e: "  << tools::to_string(pn -> e, 12) 
                  << " beta: "  << tools::to_string(pn -> beta, 12) 
                  << " mass: " << tools::to_string(pn -> mass, 12)
                  << std::endl;
        std::cout << "---------- lepton ---------- " << std::endl; 
        std::cout << " px: "  << tools::to_string(ln -> px, 12) 
                  << " py: " << tools::to_string(ln -> py, 12)
                  << " pz: " << tools::to_string(ln  -> pz, 12) 
                  << " e: "  << tools::to_string(ln -> e, 12) 
                  << " beta: " << tools::to_string(ln -> beta, 12) 
                  << " mass: " << tools::to_string(ln -> mass, 12)
                  << std::endl; 
 
        std::cout << "---------- bquark ---------- " << std::endl; 
        std::cout << " px: " << tools::to_string(bn -> px, 12) 
                  << " py: " << tools::to_string(bn -> py, 12)
                  << " pz: " << tools::to_string(bn -> pz, 12) 
                  << " e: "  << tools::to_string(bn -> e, 12) 
                  << " beta: "  << tools::to_string(bn -> beta, 12) 
                  << " mass: " << tools::to_string(bn -> mass, 12)
                  << std::endl;
        conuic c = conuic(bs_[itn -> first], ls_[itn -> first]); 
        c.proof(pn); 
    }
    return false; 

    
    std::cout << "(nu-nus) -> " << nus_.size() << std::endl; 
    // ------ get the detector level objects ------ //
    std::map<int, std::vector<particle_template*>> jets_; 
    std::map<int, std::vector<particle_template*>> lept_; 
    for (int i(0); i < detectors.size(); ++i){
        particle_template* p = detectors[i];
        std::string type = p -> type;

        std::map<int, std::vector<particle_template*>>* mx = nullptr;
        if (type == "jet"){mx = &jets_;}
        else {mx = &lept_;}

        std::vector<int> top_idx = {};
        if (type == "jet"){top_idx = p -> as<jet>() -> top_index;}
        if (type == "el"){top_idx = {p -> as<electron>() -> top_index};}
        if (type == "mu"){top_idx = {p -> as<muon>() -> top_index};}
        for (int k(0); k < top_idx.size(); ++k){
            if (!nus_[top_idx[k]]){continue;}
            (*mx)[top_idx[k]].push_back(p);
        } 
    }
    
    std::map<int, std::vector<particle_template*>>::iterator itj; 
    std::map<int, std::vector<particle_template*>>::iterator itl; 
    for (itj = jets_.begin(); itj != jets_.end(); ++itj){
        for (itl = lept_.begin(); itl != lept_.end(); ++itl){
            if (!itl -> second.size()){continue;}
            particle_template* jtsx = nullptr; 
            this -> sum(&itj -> second, &jtsx); 

            particle_template* lts = itl -> second[0]; 
            conuic(jtsx, lts); 
        }
    }

    abort(); 
    return true; 
}



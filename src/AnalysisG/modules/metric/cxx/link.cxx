#include <templates/metric_template.h>
#include <templates/model_template.h>
#include <structs/switchboards.h>

std::vector<particle_template*> metric_template::make_particle(
        std::vector<std::vector<double>>* pt , std::vector<std::vector<double>>* eta, 
        std::vector<std::vector<double>>* phi, std::vector<std::vector<double>>* energy
){
    std::vector<particle_template*> ptx(pt -> size(), nullptr); 
    for (size_t x(0); x < pt -> size(); ++x){
        particle_template* px = new particle_template(); 
        px -> index = x; 
        px -> pt    = pt  -> at(x)[0]; 
        px -> eta   = eta -> at(x)[0]; 
        px -> phi   = phi -> at(x)[0]; 
        px -> e     = energy -> at(x)[0]; 
        px -> _is_marked = true; 
        this -> garbage[px -> hash].push_back(px); 
        ptx[x] = px; 
    }    
    return ptx; 
}

std::map<int, torch::TensorOptions*> metric_template::get_devices(){
    std::map<int, torch::TensorOptions*> out = {}; 
    std::map<std::string, model_template*>::iterator itx; 
    for (itx = this -> lnks.begin(); itx != this -> lnks.end(); ++itx){
        int dx = itx -> second -> device_index; 
        if (out[dx]){continue;}
        out[dx] = itx -> second -> m_option; 
    }
    return out;
}

void metric_template::flush_garbage(){
    std::map<std::string, std::vector<particle_template*>>::iterator itr;
    for (itr = this -> garbage.begin(); itr != this -> garbage.end(); ++itr){
        for (size_t x(0); x < itr -> second.size(); ++x){
            if (!itr -> second[x] -> _is_marked){continue;}
            delete itr -> second[x];
        }
        itr -> second.clear(); 
    }
    this -> garbage.clear();
}

bool metric_template::link(model_template* mdl){
    std::string mdlx = mdl -> name; 
    if (this -> lnks.count(mdlx)){return false;}
    this -> info("Linking Model: " + mdlx + " to metric: " + std::string(this -> name)); 
    this -> lnks[mdlx] = mdl; 
   
    for (size_t x(0); x < this -> data -> size(); ++x){
        this -> data -> at(x) -> model = mdl; 
        this -> data -> at(x) -> dev   = mdl -> m_option; 
    }
    return true; 
}



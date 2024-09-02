#include "gnn-particles.h"

particle_gnn::particle_gnn(){
    this -> type = "gnn-particle"; 
    this -> add_leaf("pt", "pt"); 
    this -> add_leaf("eta", "eta"); 
    this -> add_leaf("phi", "phi"); 
    this -> add_leaf("energy", "energy"); 
    this -> add_leaf("lep", "is_lep"); 
}

particle_gnn::~particle_gnn(){}

particle_template* particle_gnn::clone(){return (particle_template*)new particle_gnn();}

void particle_gnn::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<std::vector<double>> _pt, _eta, _phi, _energy; 
    el -> get("pt"    , &_pt); 
    el -> get("eta"   , &_eta); 
    el -> get("phi"   , &_phi); 
    el -> get("energy", &_energy); 

    std::vector<int> _is_lep, _is_b; 
    el -> get("lep", &_is_lep); 
    for (int x(0); x < _energy.size(); ++x){
        particle_gnn* p = new particle_gnn(); 
        p -> index      = x; 
        p -> pt         = _pt[x][0]; 
        p -> eta        = _eta[x][0]; 
        p -> phi        = _phi[x][0]; 
        p -> e          = _energy[x][0]; 
        p -> is_lep     = _is_lep[x]; 
        (*prt)[std::string(p -> hash)] = p; 
    }
}

top::top(){
    this -> type = "top"; 
}
top::~top(){}

zprime::zprime(){
    this -> type = "zprime"; 
}

zprime::~zprime(){}


#include "gnn-particles.h"

particle_gnn::particle_gnn(){
    this -> type = "gnn-particle"; 
    this -> add_leaf("pt", "pt"); 
    this -> add_leaf("eta", "eta"); 
    this -> add_leaf("phi", "phi"); 
    this -> add_leaf("energy", "energy"); 
}

particle_gnn::~particle_gnn(){}

particle_template* particle_gnn::clone(){return (particle_template*)new particle_gnn();}

void particle_gnn::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<std::vector<double>> _pt, _eta, _phi, _energy; 
    el -> get("pt"    , &_pt); 
    el -> get("eta"   , &_eta); 
    el -> get("phi"   , &_phi); 
    el -> get("energy", &_energy); 

    for (int x(0); x < _energy.size(); ++x){
        particle_gnn* p = new particle_gnn(); 
        p -> index      = x; 
        p -> pt         = _pt[x][0]; 
        p -> eta        = _eta[x][0]; 
        p -> phi        = _phi[x][0]; 
        p -> e          = _energy[x][0]; 
        (*prt)[std::string(p -> hash)] = p; 
    }
}

top_gnn::top_gnn(){
    this -> type = "gnn-top"; 
    this -> add_leaf("pmc", "top_pmc"); 
}

top_gnn::~top_gnn(){}

particle_template* top_gnn::clone(){return (particle_template*)new top_gnn();}

void top_gnn::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<std::vector<float>> _pmc; 
    el -> get("pmc", &_pmc); 

    for (int x(0); x < _pmc.size(); ++x){
        top_gnn* p = new top_gnn(); 
        p -> index = x; 
        p -> px    = _pmc[x][0]; 
        p -> py    = _pmc[x][1]; 
        p -> pz    = _pmc[x][2]; 
        p -> e     = _pmc[x][3]; 
        (*prt)[std::string(p -> hash)] = p; 
    }
}


top_truth::top_truth(){
    this -> type = "truth-top"; 
    this -> add_leaf("pmc", "truth_top_pmc"); 
}

top_truth::~top_truth(){}

particle_template* top_truth::clone(){return (particle_template*)new top_truth();}

void top_truth::build(std::map<std::string, particle_template*>* prt, element_t* el){
    std::vector<std::vector<float>> _pmc; 
    el -> get("pmc", &_pmc); 

    for (int x(0); x < _pmc.size(); ++x){
        top_truth* p = new top_truth(); 
        p -> index   = x; 
        p -> px      = _pmc[x][0]; 
        p -> py      = _pmc[x][1]; 
        p -> pz      = _pmc[x][2]; 
        p -> e       = _pmc[x][3]; 
        (*prt)[std::string(p -> hash)] = p; 
    }
}


zprime::zprime(){
    this -> type = "zprime"; 
}

zprime::~zprime(){}

particle_template* zprime::clone(){return (particle_template*)new zprime();}

void zprime::build(std::map<std::string, particle_template*>* prt, element_t* el){}


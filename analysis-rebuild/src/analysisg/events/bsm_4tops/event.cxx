#include "event.h"

bsm_4tops::bsm_4tops(){
    this -> name = "bsm_4tops"; 
    this -> add_leaf("weight", "weight_mc"); 
    this -> add_leaf("event_number", "eventNumber"); 
    this -> add_leaf("mu", "mu"); 
    this -> add_leaf("met", "met_met"); 
    this -> add_leaf("phi", "met_phi"); 
    this -> trees = {"nominal"}; 

    this -> register_particle(&this -> Tops);
    this -> register_particle(&this -> Children); 
    this -> register_particle(&this -> TruthJets);
    this -> register_particle(&this -> Jets); 
    this -> register_particle(&this -> Partons); 
}

bsm_4tops::~bsm_4tops(){}

event_template* bsm_4tops::clone(){return (event_template*)new bsm_4tops();}

void bsm_4tops::build(element_t* el){
    el -> get("event_number", &this -> event_number); 

}


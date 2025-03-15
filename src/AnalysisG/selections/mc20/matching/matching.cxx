#include "matching.h"

selection_template* matching::clone(){return (selection_template*)new matching();}
matching::~matching(){
    //auto flush =[](std::vector<particle*>* inpt){
    //    for (size_t x(0); x < inpt -> size(); ++x){delete (*inpt)[x];}
    //}; 

    //for (size_t x(0); x < this -> output.size(); ++x){
    //    flush(&this -> output[x].truth_tops   ); 
    //    flush(&this -> output[x].children_tops); 
    //    flush(&this -> output[x].truth_jets   ); 
    //    flush(&this -> output[x].jets_children); 
    //    flush(&this -> output[x].jets_leptons ); 
    //} 
}

matching::matching(){this -> name = "matching";}

void matching::merge(selection_template* sl){
    matching* slt = (matching*)sl; 
    merge_data(&this -> output, &slt -> output); 
    slt -> output.clear(); 
}

bool matching::strategy(event_template* ev){
    std::string evnt = ev -> name; 
    if (evnt == "experimental_mc20"){this -> experimental(ev);}
    if (evnt == "ssml_mc20"){this -> current(ev);}
    if (evnt == "bsm_4tops"){this -> reference(ev);}
    return true; 
}

void matching::collect(std::vector<particle_template*>* data, std::vector<particle*>* out, std::string hash_top){
    if (!data -> size()){return;}
    particle_template* ptk = nullptr; 
    this -> sum(data, &ptk);
    particle* drv = new particle(ptk, true); 
    drv -> root_hash = hash_top; 
    out -> push_back(drv); 
}


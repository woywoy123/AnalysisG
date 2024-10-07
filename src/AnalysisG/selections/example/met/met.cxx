#include "met.h"

met::met(){this -> name = "met";}
met::~met(){}

selection_template* met::clone(){
    return (selection_template*)new met();
}

void met::merge(selection_template* sl){
    met* slt = (met*)sl; 
    merge_data(&this -> missing_et, &slt -> missing_et); 
}

bool met::selection(event_template* ev){return true;}
bool met::strategy(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 
    this -> missing_et[evn -> hash] = evn -> met; 
    return true; 
}


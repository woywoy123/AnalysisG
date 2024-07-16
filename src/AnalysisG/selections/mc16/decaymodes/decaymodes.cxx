#include "decaymodes.h"

decaymodes::decaymodes(){this -> name = "decaymodes";}
decaymodes::~decaymodes(){}

selection_template* decaymodes::clone(){
    return (selection_template*)new decaymodes();
}

void decaymodes::merge(selection_template* sl){
    decaymodes* slt = (decaymodes*)sl; 

    // example variable
    //merge_data(&this -> <var-name>, &slt -> <var-name>); 
}

bool decaymodes::selection(event_template* ev){return true;}

bool decaymodes::strategy(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 

    return true; 
}


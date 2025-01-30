#include "event.h"

event::event(){this -> name = "event";}
event::~event(){}

selection_template* event::clone(){
    return (selection_template*)new event();
}

void event::merge(selection_template* sl){
    event* slt = (event*)sl; 

    // example variable
//    merge_data(&this -> <var-name>, &slt -> <var-name>); 
}

bool event::selection(event_template* ev){return true;}

bool event::strategy(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 
    return true; 
}


#include "topefficiency.h"

topefficiency::topefficiency(){this -> name = "topefficiency";}
topefficiency::~topefficiency(){}

selection_template* topefficiency::clone(){
    return (selection_template*)new topefficiency();
}

void topefficiency::merge(selection_template* sl){
    topefficiency* slt = (topefficiency*)sl; 

    // example variable
    //merge_data(&this -> <var-name>, &slt -> <var-name>); 
}

bool topefficiency::selection(event_template* ev){return true;}

bool topefficiency::strategy(event_template* ev){
    bsm_4tops* evn = (bsm_4tops*)ev; 

    return true; 
}


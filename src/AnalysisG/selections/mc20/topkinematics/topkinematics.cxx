#include "topkinematics.h"

topkinematics::topkinematics(){this -> name = "topkinematics";}
topkinematics::~topkinematics(){}

selection_template* topkinematics::clone(){
    return (selection_template*)new topkinematics();
}

void topkinematics::merge(selection_template* sl){
    topkinematics* slt = (topkinematics*)sl; 

    //merge_data(&this -> <var-name>, &slt -> <var-name>); 
}

bool topkinematics::selection(event_template* ev){
    ssml_mc20* evn = (ssml_mc20*)ev; 
    std::vector<particle_template*>* tops = &evn -> Tops; 
    int res = 0; 
    for (size_t x(0); x < tops -> size(); ++x){res += ((top*)tops -> at(x)) -> from_res;}
    std::cout << res << std::endl; 
    abort(); 
    return res == 2;
}

bool topkinematics::strategy(event_template* ev){
    //<event-name>* evn = (<event-name>*)ev; 

    return true; 
}


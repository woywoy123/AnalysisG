#include <generators/eventgenerator.h>

eventgenerator::eventgenerator(){
    this -> prefix = "event-generator"; 
}

eventgenerator::~eventgenerator(){}

void eventgenerator::add_event_template(std::map<std::string, event_template*>* inpt){
    this -> add_event(inpt);
}

void eventgenerator::flush_events(){
    std::map<std::string, container*>::iterator itr = this -> root_container -> begin(); 
    for (; itr != this -> root_container -> end(); ++itr){itr -> second -> flush_events();}
}

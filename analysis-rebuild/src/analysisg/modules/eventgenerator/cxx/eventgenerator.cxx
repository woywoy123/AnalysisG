#include <generators/eventgenerator.h>

eventgenerator::eventgenerator(){
    this -> prefix = "event-generator"; 
}

eventgenerator::~eventgenerator(){}

void eventgenerator::add_event_template(std::map<std::string, event_template*>* inpt){
    this -> add_event(inpt);
}


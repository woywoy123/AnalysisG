#include <generators/graphgenerator.h>

graphgenerator::graphgenerator(){
    this -> prefix = "graph-generator"; 
}

graphgenerator::~graphgenerator(){}

void graphgenerator::add_graph_template(std::map<std::string, graph_template*>* inpt){
    this -> add_event(inpt);
}


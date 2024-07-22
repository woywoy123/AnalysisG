#include "ssml_mc20.h"

ssml_mc20::ssml_mc20(){
    this -> name = "ssml_mc20"; 
    //this -> add_leaf("<key_name>", "<leaf-name>"); 
    //this -> trees = {"<tree-name>"}; 
    this -> register_particle(&this -> m_particle);
}

ssml_mc20::~ssml_mc20(){}

event_template* ssml_mc20::clone(){return (event_template*)new ssml_mc20();}

void ssml_mc20::build(element_t* el){
    //el -> get("<key_name>", &this -> key_name); 
}

void ssml_mc20::CompileEvent(){
    //std::map<std::string, some_particle*> particle = this -> m_particle; 

    // do some stuff here
}

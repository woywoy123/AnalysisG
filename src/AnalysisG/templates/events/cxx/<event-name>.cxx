#include "<event-name>.h"

<event-name>::<event-name>(){
    this -> name = "<event-name>"; 
    this -> add_leaf("<key_name>", "<leaf-name>"); 
    this -> trees = {"<tree-name>"}; 
    this -> register_particle(&this -> m_particle);
}

<event-name>::~<event-name>(){}

event_template* <event-name>::clone(){return (event_template*)new <event-name>();}

void <event-name>::build(element_t* el){
    el -> get("<key_name>", &this -> key_name); 
}

void <event-name>::CompileEvent(){
    std::map<std::string, some_particle*> particle = this -> m_particle; 

    // do some stuff here
}

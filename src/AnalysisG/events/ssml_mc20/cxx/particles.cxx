#include "particles.h"

<particle-name>::<particle-name>(){
    this -> type = "<some-particle>"; 
    this -> add_leaf("<kinematic-key>", "<leaf-name-of-kinematic>"); 

    // adds <some-particle>/<leaf-name-of-kinematic>
    // to the search when looking at leaves. 
    this -> apply_type_prefix(); 
}

<particle-name>::~<particle-name>(){}

particle_template* <particle-name>::clone(){return (event_template*)new <particle-name>();}

void build(std::map<std::string, particle_template*>* prt, element_t* el){
    // here the framework builds the particle and assigns values to the object.
    
    std::vector<float> some_kinematic = {}; 
    el -> get("<kinematic-key>", &some_kinematic); 
    
    for (int x(0); x < some_kinematic.size(); ++x){
        <particle-name>* p = new <particle-name>(); 
        p -> key_variable = some_kinematic[x]; 

        // IMPORTANT! Make the particle available to the event.
        (*prt)[std::string(p -> hash)] = p; 
    }
}

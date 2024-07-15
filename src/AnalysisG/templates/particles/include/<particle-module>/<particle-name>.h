#ifndef EVENTS_<PARTICLE-NAME>_H
#define EVENTS_<PARTICLE-NAME>_H

#include <templates/particle_template.h>

class <particle-name>: public particle_template
{
    public:
        <particle-name>(); 
        ~<particle-name>() override; 

        float key_variable = 0; // some variable of the particle

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 

}; 


#endif

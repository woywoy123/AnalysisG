#ifndef EVENTS_SSML_MC20_H
#define EVENTS_SSML_MC20_H

//#include <<particle-module>/<particle-header>.h>
#include <templates/event_template.h>

class ssml_mc20: public event_template
{
    public:
        ssml_mc20(); 
        ~ssml_mc20() override; 

        //std::vector<particle_template*> some_particles = {}; 

        //float key_name = 0; // some variable of the event 

        event_template* clone() override; 
        void build(element_t* el) override; 
        void CompileEvent() override; 

    //private: 
        //std::map<std::string, some_particle*> m_particle = {}; 
}; 


#endif

#ifndef EVENTS_<EVENT-NAME>_H
#define EVENTS_<EVENT-NAME>_H

#include <<particle-module>/<particle-header>.h>
#include <templates/event_template.h>

class <event-name>: public event_template
{
    public:
        <event-name>(); 
        ~<event-name>() override; 

        std::vector<particle_template*> some_particles = {}; 

        float key_name = 0; // some variable of the event 

        event_template* clone() override; 
        void build(element_t* el) override; 
        void CompileEvent() override; 

    private: 
        std::map<std::string, some_particle*> m_particle = {}; 
}; 


#endif

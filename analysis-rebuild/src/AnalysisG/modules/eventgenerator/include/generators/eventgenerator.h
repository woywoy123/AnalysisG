#ifndef EVENT_GENERATOR_H
#define EVENT_GENERATOR_H

#include <generators/sampletracer.h>

class eventgenerator: public sampletracer 
{
    public:
        eventgenerator();
        ~eventgenerator();

        void add_event_template(std::map<std::string, event_template*>* inpt); 
        void flush_events(); 
}; 

#endif

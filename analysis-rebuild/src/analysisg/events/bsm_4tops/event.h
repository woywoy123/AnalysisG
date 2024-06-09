#ifndef EVENTS_BSM4TOPS_H
#define EVENTS_BSM4TOPS_H

#include "particles.h"
#include <templates/event_template.h>

class bsm_4tops: public event_template
{
    public:
        bsm_4tops(); 
        ~bsm_4tops(); 

        std::map<std::string, top*> Tops = {}; 
        std::map<std::string, children*> Children = {}; 
        std::map<std::string, truthjet*> TruthJets = {}; 
        std::map<std::string, jet*> Jets = {}; 
        std::map<std::string, parton*> Partons = {};  

        unsigned long long event_number = 0; 
        double mu = 0; 
        double met = 0; 
        double phi = 0; 

        event_template* clone() override; 
        void build(element_t* el) override; 

}; 


#endif

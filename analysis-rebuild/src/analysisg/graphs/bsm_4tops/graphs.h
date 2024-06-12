#ifndef GRAPHS_BSM4TOPS_H
#define GRAPHS_BSM4TOPS_H

#include <bsm_4tops/event.h>
#include <templates/graph_template.h>
#include "features.h"

class truth_tops: public graph_template
{
    public:
        truth_tops(); 
        ~truth_tops() override; 
        graph_template* clone() override; 

        void build_event(event_template* ev) override; 

}; 


#endif

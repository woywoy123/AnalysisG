#ifndef GRAPHS_BSM4TOPS_H
#define GRAPHS_BSM4TOPS_H

#include <bsm_4tops/event.h>
#include <templates/graph_template.h>
#include "features.h"

class graph_tops: public graph_template
{
    public:
        graph_tops(); 
        ~graph_tops() override; 
        graph_template* clone() override; 
        void CompileEvent() override; 
}; 

class graph_children: public graph_template
{
    public:
        graph_children(); 
        ~graph_children() override; 
        graph_template* clone() override; 
        void CompileEvent() override; 
}; 


#endif

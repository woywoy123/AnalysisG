#ifndef GRAPHS_EXP_MC20_H
#define GRAPHS_EXP_MC20_H

#include <templates/graph_template.h>
#include <exp_mc20/event.h>

class graph_jets: public graph_template
{
    public:
        graph_jets(); 
        ~graph_jets() override; 
        graph_template* clone() override; 
        void CompileEvent() override; 
}; 


class graph_jets_nonu: public graph_template
{
    public:
        graph_jets_nonu(); 
        ~graph_jets_nonu() override; 
        graph_template* clone() override; 
        void CompileEvent() override; 
}; 

class graph_jets_detector_lep: public graph_template
{
    public:
        graph_jets_detector_lep(); 
        ~graph_jets_detector_lep() override; 
        graph_template* clone() override; 
        void CompileEvent() override; 
}; 

class graph_detector: public graph_template
{
    public:
        graph_detector(); 
        ~graph_detector() override; 
        graph_template* clone() override; 
        void CompileEvent() override; 
}; 


#endif

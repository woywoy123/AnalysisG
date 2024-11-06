#ifndef GRAPHS_BSM4TOPS_H
#define GRAPHS_BSM4TOPS_H

#include <templates/graph_template.h>
#include <bsm_4tops/event.h>

class graph_tops: public graph_template
{
    public:
        graph_tops(); 
        ~graph_tops(); 
        graph_template* clone() override; 
        void CompileEvent() override; 
}; 

class graph_children: public graph_template
{
    public:
        graph_children(); 
        ~graph_children(); 
        graph_template* clone() override; 
        void CompileEvent() override; 
}; 

class graph_truthjets: public graph_template
{
    public:
        graph_truthjets(); 
        ~graph_truthjets(); 
        graph_template* clone() override; 
        void CompileEvent() override; 
}; 

class graph_truthjets_nonu: public graph_template
{
    public:
        graph_truthjets_nonu(); 
        ~graph_truthjets_nonu(); 
        graph_template* clone() override; 
        void CompileEvent() override; 
}; 

class graph_jets: public graph_template
{
    public:
        graph_jets(); 
        ~graph_jets(); 
        graph_template* clone() override; 
        void CompileEvent() override; 
}; 


class graph_jets_nonu: public graph_template
{
    public:
        graph_jets_nonu(); 
        ~graph_jets_nonu(); 
        graph_template* clone() override; 
        void CompileEvent() override; 
}; 

class graph_jets_detector_lep: public graph_template
{
    public:
        graph_jets_detector_lep(); 
        ~graph_jets_detector_lep(); 
        graph_template* clone() override; 
        void CompileEvent() override; 
}; 

class graph_detector: public graph_template
{
    public:
        graph_detector(); 
        ~graph_detector(); 
        graph_template* clone() override; 
        void CompileEvent() override; 
}; 


#endif

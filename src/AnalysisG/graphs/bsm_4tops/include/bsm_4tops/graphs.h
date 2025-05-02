#ifndef GRAPHS_BSM4TOPS_H
#define GRAPHS_BSM4TOPS_H

#include <templates/graph_template.h>

class graph_tops: public graph_template
{
    public:
        graph_tops(); 
        virtual ~graph_tops(); 
        graph_template* clone() override; 
        void CompileEvent() override; 
}; 

class graph_children: public graph_template
{
    public:
        graph_children(); 
        virtual ~graph_children(); 
        graph_template* clone() override; 
        void CompileEvent() override; 
}; 

class graph_truthjets: public graph_template
{
    public:
        graph_truthjets(); 
        virtual ~graph_truthjets(); 
        graph_template* clone() override; 
        void CompileEvent() override; 
}; 

class graph_truthjets_nonu: public graph_template
{
    public:
        graph_truthjets_nonu(); 
        graph_template* clone() override; 
        virtual ~graph_truthjets_nonu(); 
        void CompileEvent() override; 
}; 

class graph_jets: public graph_template
{
    public:
        graph_jets(); 
        virtual ~graph_jets(); 
        graph_template* clone() override; 
        void CompileEvent() override; 
        bool PreSelection() override; 
}; 


class graph_jets_nonu: public graph_template
{
    public:
        graph_jets_nonu(); 
        virtual ~graph_jets_nonu(); 
        graph_template* clone() override; 
        void CompileEvent() override; 
}; 

class graph_jets_detector_lep: public graph_template
{
    public:
        graph_jets_detector_lep(); 
        virtual ~graph_jets_detector_lep(); 
        graph_template* clone() override; 
        void CompileEvent() override; 
}; 

class graph_detector: public graph_template
{
    public:
        graph_detector(); 
        virtual ~graph_detector(); 
        graph_template* clone() override; 
        void CompileEvent() override; 
        bool PreSelection() override; 
        bool force_match = false; 
        int num_cuda = 1; 
}; 


#endif

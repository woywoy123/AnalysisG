#ifndef EVENTS_PARTICLE_GNN_H
#define EVENTS_PARTICLE_GNN_H

#include <templates/particle_template.h>

class particle_gnn: public particle_template
{
    public:
        particle_gnn(); 
        ~particle_gnn() override; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 
}; 

class top_gnn: public particle_template
{
    public:
        top_gnn(); 
        ~top_gnn() override; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 
}; 

class top_truth: public particle_template
{
    public:
        top_truth(); 
        ~top_truth() override; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 
}; 


class zprime: public particle_template
{
    public:
        zprime(); 
        ~zprime() override; 
        std::vector<top_gnn*> matched_tops; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 
}; 


#endif

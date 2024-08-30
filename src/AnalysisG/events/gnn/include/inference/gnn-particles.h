#ifndef EVENTS_PARTICLE_GNN_H
#define EVENTS_PARTICLE_GNN_H

#include <templates/particle_template.h>

class particle_gnn: public particle_template
{
    public:
        particle_gnn(); 
        ~particle_gnn() override; 
        bool is_lep = false; 

        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 
}; 

class top: public particle_template
{
    public:
        top(); 
        ~top() override; 
        bool is_lep = false; 
        bool is_truth = false;
}; 


class zprime: public particle_template
{
    public:
        zprime(); 
        ~zprime() override; 
        bool is_truth = false; 
}; 


#endif

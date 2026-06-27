#ifndef EVENTS_PARTICLE_GNN_H
#define EVENTS_PARTICLE_GNN_H

#include <templates/particle_template.h>
enum class pagerank_e; 

class particle_gnn: public particle_template
{
    public:
        particle_gnn(); 
        ~particle_gnn() override; 
        bool lep = false; 
        
        std::map<pagerank_e, float> pr_score = {}; 
        particle_template* clone() override; 
        void build(std::map<std::string, particle_template*>* prt, element_t* el) override; 
}; 

class top: public particle_template
{
    public:
        top(); 
        ~top() override; 
        float av_score = 0;
        int n_leps = 0;  
        int n_nodes = 0; 

        pagerank_e mode; 
}; 


class zprime: public particle_template
{
    public:
        zprime(); 
        ~zprime() override; 
        float av_score = 0; 
        int n_leps = 0;  
        int n_nodes = 0; 
        
        pagerank_e mode; 
}; 


#endif

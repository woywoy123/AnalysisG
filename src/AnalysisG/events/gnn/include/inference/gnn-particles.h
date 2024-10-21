#ifndef EVENTS_PARTICLE_GNN_H
#define EVENTS_PARTICLE_GNN_H

#include <templates/particle_template.h>

template <typename g>
void reduce(element_t* el, std::string key, std::vector<g>* out);

template <typename g>
void reduce(element_t* el, std::string key, g* out);

template <typename g>
void read(element_t* el, std::string key, std::vector<g>* out);

class particle_gnn: public particle_template
{
    public:
        particle_gnn(); 
        ~particle_gnn() override; 
        bool lep = false; 

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
}; 


class zprime: public particle_template
{
    public:
        zprime(); 
        ~zprime() override; 
        float av_score = 0; 
        int n_leps = 0;  
        int n_nodes = 0; 
}; 


#endif

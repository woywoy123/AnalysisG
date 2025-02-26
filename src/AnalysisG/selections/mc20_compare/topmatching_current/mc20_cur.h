#ifndef MC20_CURRENT_H
#define MC20_CURRENT_H

#include <ssml_mc20/event.h>
#include <templates/selection_template.h>

class particle: public particle_template
{
    public:
        using particle_template::particle_template;
}; 

struct packet_t {
    std::vector<particle*> truth_tops;
    std::vector<particle*> children_tops; 
    std::vector<particle*> truth_jets; 
    std::vector<particle*> jets_children; 
    std::vector<particle*> jets_leptons; 
}; 

class mc20_current: public selection_template
{
    public:
        mc20_current();
        selection_template* clone() override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;
        std::vector<packet_t> output; 
};

#endif

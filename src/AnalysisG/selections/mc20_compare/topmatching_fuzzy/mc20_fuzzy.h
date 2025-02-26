#ifndef MC20_FUZZY_H
#define MC20_FUZZY_H

#include <exp_mc20/event.h>
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

class mc20_fuzzy: public selection_template
{
    public:
        mc20_fuzzy();
        ~mc20_fuzzy() override; 
        selection_template* clone() override; 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;

        std::vector<packet_t> output; 
};

#endif

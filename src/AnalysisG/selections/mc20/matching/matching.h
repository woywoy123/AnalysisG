#ifndef MATCHING_H
#define MATCHING_H

#include <templates/selection_template.h>

class particle: public particle_template
{
    public:
        using particle_template::particle_template;
        std::string root_hash = ""; 
}; 

struct packet_t {
    std::vector<particle*> truth_tops;
    std::vector<particle*> children_tops; 
    std::vector<particle*> truth_jets; 
    std::vector<particle*> jets_children; 
    std::vector<particle*> jets_leptons; 
}; 

class matching: public selection_template
{
    public:
        matching();
        ~matching(); 
        selection_template* clone() override; 
       
        void collect(std::vector<particle_template*>* data, std::vector<particle*>* out, std::string hash_top); 

        void reference(event_template* ev);
        void experimental(event_template* ev); 
        void current(event_template* ev); 

        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;
        std::vector<packet_t> output; 
};

#endif

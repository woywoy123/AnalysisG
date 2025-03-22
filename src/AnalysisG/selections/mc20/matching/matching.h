#ifndef MATCHING_H
#define MATCHING_H
#include <templates/selection_template.h>

struct object_data_t {
    int num_tops = 0; 
    int num_ltop = 0; 
    int num_htop = 0; 
    std::vector<double> mass = {}; 

    std::vector<int> num_jets = {}; 
    std::vector<int> is_leptonic = {}; 
    std::vector<int> is_hadronic = {}; 
    std::vector<std::vector<int>> merged = {}; 
    std::vector<std::vector<int>> pdgid = {}; 
}; 

struct buffer_t {
    object_data_t top_partons;
    object_data_t top_children; 
    object_data_t top_truthjets; 
    object_data_t top_jets_children; 
    object_data_t top_jets_leptons; 
}; 

class matching: public selection_template
{
    public:
        matching();
        ~matching(); 
        selection_template* clone() override; 
       
        void reference(event_template* ev);
        void experimental(event_template* ev); 
        void current(event_template* ev); 

        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;

        std::vector<int> get_pdgid(std::vector<particle_template*>* prt); 


        buffer_t data; 
};

#endif

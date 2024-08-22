#ifndef ZPRIME_H
#define ZPRIME_H

#include <bsm_4tops/event.h>
#include <templates/selection_template.h>

class zprime: public selection_template
{
    public:
        zprime();
        ~zprime() override; 
        selection_template* clone() override; 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;

        std::vector<float> zprime_truth_tops = {}; 
        std::vector<float> zprime_children = {}; 
        std::vector<float> zprime_truthjets = {}; 
        std::vector<float> zprime_jets = {}; 
};

#endif

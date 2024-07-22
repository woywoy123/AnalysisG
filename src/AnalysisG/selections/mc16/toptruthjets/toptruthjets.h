#ifndef TOPTRUTHJETS_H
#define TOPTRUTHJETS_H

#include <bsm_4tops/event.h>
#include <templates/selection_template.h>

class toptruthjets: public selection_template
{
    public:
        toptruthjets();
        ~toptruthjets() override; 
        selection_template* clone() override; 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;

        std::map<std::string, std::map<std::string, std::map<std::string, std::vector<float>>>> top_mass = {}; 
        std::map<std::string, std::map<std::string, std::map<std::string, std::vector<float>>>> truthjet_partons = {}; 
        std::map<std::string, std::map<std::string, std::map<std::string, std::vector<float>>>> truthjets_contribute = {}; 
        std::map<std::string, std::map<std::string, std::vector<float>>> truthjet_top = {}; 
        std::map<std::string, std::vector<float>> truthjet_mass = {}; 

        std::vector<int> ntops_lost = {}; 

};

#endif

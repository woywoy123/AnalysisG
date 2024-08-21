#ifndef TOPJETS_H
#define TOPJETS_H

#include <bsm_4tops/event.h>
#include <templates/selection_template.h>

class topjets: public selection_template
{
    public:
        topjets();
        ~topjets() override; 
        selection_template* clone() override; 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;

        std::map<std::string, std::map<std::string, std::map<std::string, std::vector<float>>>> top_mass = {}; 
        std::map<std::string, std::map<std::string, std::map<std::string, std::vector<float>>>> jet_partons = {}; 
        std::map<std::string, std::map<std::string, std::map<std::string, std::vector<float>>>> jets_contribute = {}; 
        std::map<std::string, std::map<std::string, std::vector<float>>> jet_top = {}; 
        std::map<std::string, std::vector<float>> jet_mass = {}; 
        std::vector<int> ntops_lost = {}; 
};

#endif

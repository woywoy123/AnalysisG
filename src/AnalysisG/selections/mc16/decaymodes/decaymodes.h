#ifndef SELECTION_DECAYMODES_H
#define SELECTION_DECAYMODES_H

#include <bsm_4tops/event.h>
#include <templates/selection_template.h>

class decaymodes: public selection_template
{
    public:
        decaymodes();
        ~decaymodes() override; 
        selection_template* clone() override; 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;

        std::map<std::string, std::vector<double>> res_top_modes = {}; 
        std::map<std::string, std::vector<double>> res_top_charges = {}; 
        std::map<std::string, int> res_top_pdgid = {}; 

        std::map<std::string, std::vector<double>> spec_top_modes = {}; 
        std::map<std::string, std::vector<double>> spec_top_charges = {}; 
        std::map<std::string, int> spec_top_pdgid = {}; 

        std::map<std::string, int> all_pdgid = {}; 
        std::map<std::string, std::vector<double>> signal_region = {}; 
        std::map<std::string, int> lepton_statistics = {}; 
        std::vector<int> ntops = {};  
};

#endif

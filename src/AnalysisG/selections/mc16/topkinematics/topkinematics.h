#ifndef MC16_TOPKINEMATICS_H
#define MC16_TOPKINEMATICS_H

#include <bsm_4tops/event.h>
#include <templates/selection_template.h>

class topkinematics: public selection_template
{
    public:
        topkinematics();
        ~topkinematics() override;
        selection_template* clone() override;

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override; 
        void merge(selection_template* sl) override; 

        std::map<std::string, std::vector<float>> res_top_kinematics = {};
        std::map<std::string, std::vector<float>> spec_top_kinematics = {}; 
        std::map<std::string, std::vector<float>> mass_combi = {}; 
        std::map<std::string, std::vector<float>> deltaR = {};

}; 

#endif

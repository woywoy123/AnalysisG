#ifndef REGIONS_H
#define REGIONS_H

#include <ssml_mc20/event.h>
#include <templates/selection_template.h>

struct regions_t {
    double weight = 0; 
    double variable1 = 0;
    double variable2 = 0;  
    bool passed = true; 
};

struct package_t {
    regions_t CRttbarCO2l_CO; 
    regions_t CRttbarCO2l_CO_2b; 
    regions_t CRttbarCO2l_gstr; 
    regions_t CRttbarCO2l_gstr_2b; 
    regions_t CR1b3lem; 
    regions_t CR1b3le; 
    regions_t CR1b3lm; 
    regions_t CRttW2l_plus; 
    regions_t CRttW2l_minus; 
    regions_t CR1bplus; 
    regions_t CR1bminus; 
    regions_t CRttW2l; 
    regions_t VRttZ3l; 
    regions_t VRttWCRSR; 
    regions_t SR4b; 
    regions_t SR2b; 
    regions_t SR3b; 
    regions_t SR2b2l; 
    regions_t SR2b3l4l; 
    regions_t SR2b4l; 
    regions_t SR3b2l; 
    regions_t SR3b3l4l; 
    regions_t SR3b4l; 
    regions_t SR4b4l; 
    regions_t SR; 
};

class regions: public selection_template
{
    public:
        regions();
        ~regions() override; 
        selection_template* clone() override; 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;

        std::vector<package_t> output;  
};

#endif

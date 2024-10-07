#ifndef MC16_MET_H
#define MC16_MET_H

#include <bsm_4tops/event.h>
#include <templates/selection_template.h>

class met: public selection_template
{
    public:
        met();
        ~met() override; 
        selection_template* clone() override; 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;
        std::map<std::string, float> missing_et; 
};

#endif

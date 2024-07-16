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


        //std::vector<float> <var-name>; 
};

#endif

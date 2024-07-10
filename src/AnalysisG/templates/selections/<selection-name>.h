#ifndef <selection-name>_H
#define <selection-name>_H

#include <<event-name>/event.h>
#include <templates/selection_template.h>

class <selection-name>: public selection_template
{
    public:
        <selection-name>();
        ~<selection-name>() override; 
        selection_template* clone() override; 

        bool selection(event_template* ev) override; 
        bool strategy(event_template* ev) override;
        void merge(selection_template* sl) override;


        std::vector<float> <var-name>; 
};

#endif

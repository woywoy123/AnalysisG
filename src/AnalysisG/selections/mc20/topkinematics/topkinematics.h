#ifndef topkinematics_H
#define topkinematics_H

#include <ssml_mc20/event.h>
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

};

#endif

#ifndef MULTISOL_H
#define MULTISOL_H
#include <templates/selection_template.h>

class multisol: public selection_template {
    public:
        multisol();
        ~multisol() override; 
        selection_template* clone() override; 
        bool selection(event_template* ev) override; 
        bool strategy(event_template*  ev) override;
        void merge(selection_template* sl) override;
};

#endif

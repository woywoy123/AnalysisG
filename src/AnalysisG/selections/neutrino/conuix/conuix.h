#ifndef CONUIX_H
#define CONUIX_H
#include <templates/selection_template.h>

class conuix: public selection_template {
    public:
        conuix();
        ~conuix() override; 
        selection_template* clone() override; 
        bool selection(event_template* ev) override; 
        bool strategy(event_template*  ev) override;
        void merge(selection_template* sl) override;
};

#endif

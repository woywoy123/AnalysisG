#ifndef METRICS_H
#define METRICS_H

#include <templates/model_template.h>


class metrics {
    public: 
        metrics(); 
        ~metrics(); 

        void register_model(model_template* model, int kfold); 



}; 

#endif

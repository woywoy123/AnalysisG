#ifndef <name>_METRIC_H
#define <name>_METRIC_H

#include <templates/metric_template.h>

class <name>_metric: public metric_template
{
    public:
        <name>_metric(); 
        ~<name>_metric() override; 
        <name>_metric* clone() override; 

        void define_metric(metric_t* mtx) override; 
        void define_variables() override; 
        void event() override; 
        void batch() override; 
        void end() override; 

    private: 
        std::string mode = ""; 
        float <name> = 0;
        float global_<name> = 0; 
}; 


#endif

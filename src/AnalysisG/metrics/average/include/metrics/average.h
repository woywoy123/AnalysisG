#ifndef AVERAGE_METRIC_H
#define AVERAGE_METRIC_H

#include <templates/metric_template.h>

class average_metric: public metric_template
{
    public:

        average_metric(); 
        ~average_metric() override; 
        average_metric* clone() override; 
}; 


#endif

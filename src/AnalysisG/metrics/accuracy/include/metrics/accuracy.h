#ifndef AVERAGE_METRIC_H
#define AVERAGE_METRIC_H

#include <templates/metric_template.h>

class accuracy_metric: public metric_template
{
    public:
        accuracy_metric(); 
        ~accuracy_metric() override; 
        accuracy_metric* clone() override; 

        void define_metric(metric_t* mtx) override; 
        void define_variables() override; 
        void event() override; 
        void batch() override; 
        void end() override; 

    private: 
        std::string mode = ""; 
        long idx = 0; 
        float edge_accuracy = 0;
        float global_edge_accuracy = 0; 

        int ntop_truth = 0; 
        std::vector<float> ntop_scores = {}; 

        std::vector<float> ntops_accuracy = {}; 
        std::map<int, int> _global_edge_accuracy = {}; 
        std::map<int, std::map<int, int>> ntop_accuracy = {}; 
}; 


#endif

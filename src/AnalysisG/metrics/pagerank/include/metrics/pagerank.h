#ifndef PAGERANK_METRIC_H
#define PAGERANK_METRIC_H

#include <templates/metric_template.h>

class pagerank_metric: public metric_template
{
    public:
        pagerank_metric(); 
        ~pagerank_metric() override; 
        pagerank_metric* clone() override; 

        void define_metric(metric_t* mtx) override; 
        void define_variables() override; 
        void event() override; 
        void batch() override; 
        void end() override; 

        void pagerank(
            std::map<int, std::map<std::string, std::string>>* out,
            std::map<int, std::map<int, float>>* bin_data
        ); 

    private: 
        std::string mode = ""; 
       // float pagerank = 0;
       // float global_pagerank = 0; 
}; 


#endif

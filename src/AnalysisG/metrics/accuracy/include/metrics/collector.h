#ifndef AVERAGE_COLLECTOR_H
#define AVERAGE_COLLECTOR_H
#include <metrics/transfer.h>
#include <metrics/particle.h>

class performance {
    
    public:
        performance(std::string mrk_, std::string mode_);
        ~performance(); 

        void compiler(); 
        performance_t* filter(std::vector<edata_t>* dfc); 

        std::map<int, std::map<int, std::vector<edata_t>>> epoch; 
        std::map<int, std::map<int, performance_t*>>       metric; 

        std::string name = ""; 
        std::string mode = ""; 

}; 

class collector : public tools
{
    public:
        collector(); 
        ~collector(); 

        void inlet(edata* data);
        void expand(edata_t* ev, pdata* px);
        std::string label(std::vector<pdata*>* vl, std::string mox);  

        // models
        std::map<std::string, performance*> training; 
        std::map<std::string, performance*> validation; 
        std::map<std::string, performance*> evaluation; 

        int epoch = -1; 
        int kfold = -1; 
        bool release = false; 
}; 


#endif

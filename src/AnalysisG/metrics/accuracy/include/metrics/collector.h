#ifndef AVERAGE_COLLECTOR_H
#define AVERAGE_COLLECTOR_H
#include <metrics/transfer.h>
#include <metrics/particle.h>

struct edata_t {
    int ntru; 
    int npre; 
    std::vector<double> pred; 
    process_t prc; 

    double acc_edges; 

    std::vector<rectop> truth;
    std::vector<rectop> nominal;
    std::vector<rectop> umasked;
    std::vector<rectop> masked; 

}; 

class performance {
    
    public:
        performance(std::string mrk_, std::string mode_);
        ~performance(); 

        std::string name = ""; 
        std::string mode = ""; 
        std::map<int, std::map<int, std::vector<edata_t>>> epoch; 
}; 

class collector 
{
    public:
        collector(); 
        ~collector(); 

        void inlet(edata* data);
        std::string label(std::vector<pdata*>* vl, std::string mox);  

        // models
        std::map<std::string, performance*> training; 
        std::map<std::string, performance*> validation; 
        std::map<std::string, performance*> evaluation; 

}; 


#endif

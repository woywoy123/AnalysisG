#ifndef REPORT_STRUCTS_H
#define REPORT_STRUCTS_H
#include <string>
#include <map>

enum mode_enum {training, validation, evaluation}; 

class metrics; 

struct model_report {
    int k;
    int epoch; 
    bool is_complete = false; 
    metrics* waiting_plot = nullptr; 

    std::map<mode_enum, std::map<std::string, float>> loss_graph = {}; 
    std::map<mode_enum, std::map<std::string, float>> loss_node = {}; 
    std::map<mode_enum, std::map<std::string, float>> loss_edge = {}; 

    std::map<mode_enum, std::map<std::string, float>> accuracy_graph = {}; 
    std::map<mode_enum, std::map<std::string, float>> accuracy_node = {}; 
    std::map<mode_enum, std::map<std::string, float>> accuracy_edge = {}; 

    std::string run_name;
    std::string mode;
   
    long iters = 0; 
    long num_evnt = 0;  
    float progress;

    std::string print(); 
    std::string prx(std::map<mode_enum, std::map<std::string, float>>* data, std::string title); 
}; 

#endif

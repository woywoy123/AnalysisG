#ifndef STRUCTS_MODEL_H
#define STRUCTS_MODEL_H
#include <structs/enums.h>
#include <string>
#include <vector>
#include <map>

struct model_settings_t {
    opt_enum    e_optim; 
    std::string s_optim; 

    std::string weight_name; 
    std::string tree_name; 

    std::string model_name; 
    std::string model_device; 
    std::string model_checkpoint_path; 
    bool inference_mode; 
    bool is_mc; 

    std::map<std::string, std::string> o_graph; 
    std::map<std::string, std::string> o_node; 
    std::map<std::string, std::string> o_edge; 

    std::vector<std::string> i_graph; 
    std::vector<std::string> i_node; 
    std::vector<std::string> i_edge; 
}; 

#endif

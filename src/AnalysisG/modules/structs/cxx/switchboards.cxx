#include <structs/switchboards.h>
#include <structs/enums.h>
#include <iostream>
#include <cstdint>
#include <sstream>
#include <vector>
#include <string>
#include <map>

std::string lower(std::string* in){
    std::string out = *in;
    for (size_t t(0); t < in -> size(); ++t){out[t] = std::tolower(out[t]);}
    return out;
}

mode_enum model_mode(std::string* val){
    std::string vl = lower(val); 
    if (vl == "training"  ){return mode_enum::training;}
    if (vl == "validation"){return mode_enum::validation;}
    if (vl == "evaluation"){return mode_enum::evaluation;} 
    return mode_enum::invalid; 
}

std::map<mode_enum, std::string> model_mode(std::map<std::string, std::string>* val){
    std::map<mode_enum, std::string> out; 
    std::map<std::string, std::string>::iterator it = val -> begin(); 
    for (; it != val -> end(); ++it){
        std::string v = it -> first; 
        mode_enum md = model_mode(&v); 
        out[md] = it -> second;
    }
    return out; 
}

std::string enums_to_string(graph_enum gr){
    switch(gr){
        case graph_enum::truth_graph:  return "truth::graph::"; 
        case graph_enum::truth_node :  return "truth::node::"; 
        case graph_enum::truth_edge :  return "truth::edge::"; 
        case graph_enum::pred_graph:   return "prediction::graph::"; 
        case graph_enum::pred_node :   return "prediction::node::"; 
        case graph_enum::pred_edge :   return "prediction::edge::"; 
        case graph_enum::data_graph:   return "data::graph::"; 
        case graph_enum::data_node :   return "data::node::"; 
        case graph_enum::data_edge :   return "data::edge::"; 
        case graph_enum::pred_extra  : return "prediction::extra::"; 
        case graph_enum::edge_index  : return "data::edge::"; 
        case graph_enum::batch_index : return "data::node:"; 
        case graph_enum::weight      : return "data::graph::"; 
        case graph_enum::batch_events: return "data::graph::"; 
        default: return "undef"; 
    }
}



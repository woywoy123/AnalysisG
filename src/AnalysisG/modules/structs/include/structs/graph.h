#ifndef GRAPH_STRUCTS_H
#define GRAPH_STRUCTS_H

#include <c10/core/DeviceType.h>
#include <iostream>
#include <string>
#include <vector>

class meta; 

struct graph_meta {
    std::string*     name = nullptr;
    std::string*     hash = nullptr;
    meta*       meta_data = nullptr; 

    long    num_nodes = 0; 
    long    index     = 0; 
    double  weight    = 1; 

    int     in_use       = 1; 
    bool    preselection = false;
    bool    is_owner     = false; 

    c10::DeviceType device = c10::kCPU; 
    std::vector<graph_meta*> batched_events = {}; 
};

#endif

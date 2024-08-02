#ifndef STRUCTS_FOLDS_H
#define STRUCTS_FOLDS_H
#include <string>

struct folds_t {
    int k = -1; 
    bool is_train = false; 
    bool is_valid = false; 
    bool is_eval = false; 
    int index; 
}; 


struct graph_hdf5 {
    int num_nodes = -1; 
    long event_index = -1;
    std::string hash; 
    std::string filename; 
    std::string edge_index; 

    std::string data_map_graph;
    std::string data_map_node;  
    std::string data_map_edge;  

    std::string truth_map_graph; 
    std::string truth_map_node;         
    std::string truth_map_edge;         

    std::string data_graph; 
    std::string data_node; 
    std::string data_edge; 

    std::string truth_graph; 
    std::string truth_node; 
    std::string truth_edge;
}; 

struct graph_hdf5_w {
    int num_nodes = -1; 
    long event_index = -1;
    char* hash; 
    char* filename; 
    char* edge_index; 

    char* data_map_graph;
    char* data_map_node;  
    char* data_map_edge;  

    char* truth_map_graph; 
    char* truth_map_node;         
    char* truth_map_edge;         

    char* data_graph; 
    char* data_node; 
    char* data_edge; 

    char* truth_graph; 
    char* truth_node; 
    char* truth_edge;
}; 


#endif

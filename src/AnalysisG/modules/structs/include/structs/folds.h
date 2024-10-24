#ifndef STRUCTS_FOLDS_H
#define STRUCTS_FOLDS_H
#include <string>

struct folds_t {
    int k = -1; 
    bool is_train = false; 
    bool is_valid = false; 
    bool is_eval = false; 
    char* hash = nullptr;
    void flush_data(){delete [] this -> hash;} 
}; 


struct graph_hdf5 {
    int    num_nodes = -1; 
    double event_weight = 1; 
    long   event_index = -1;

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
    int    num_nodes = -1; 
    double event_weight = 1; 
    long   event_index = -1;

    char* hash = nullptr; 
    char* filename = nullptr; 
    char* edge_index = nullptr; 

    char* data_map_graph = nullptr;
    char* data_map_node = nullptr;  
    char* data_map_edge = nullptr;  

    char* truth_map_graph = nullptr; 
    char* truth_map_node = nullptr;         
    char* truth_map_edge = nullptr;         

    char* data_graph = nullptr; 
    char* data_node = nullptr; 
    char* data_edge = nullptr; 

    char* truth_graph = nullptr; 
    char* truth_node = nullptr; 
    char* truth_edge = nullptr;


    void flush_data(){
        delete [] this -> hash; 
        delete [] this -> filename; 
        delete [] this -> edge_index; 

        delete [] this -> data_map_graph; 
        delete [] this -> data_map_node; 
        delete [] this -> data_map_edge; 

        delete [] this -> truth_map_graph; 
        delete [] this -> truth_map_node;
        delete [] this -> truth_map_edge; 

        delete [] this -> data_graph; 
        delete [] this -> data_node; 
        delete [] this -> data_edge; 

        delete [] this -> truth_graph; 
        delete [] this -> truth_node; 
        delete [] this -> truth_edge; 

        this -> hash = nullptr; 
        this -> filename = nullptr; 
        this -> edge_index = nullptr; 

        this -> data_map_graph = nullptr; 
        this -> data_map_node = nullptr; 
        this -> data_map_edge = nullptr; 

        this -> truth_map_graph = nullptr; 
        this -> truth_map_node = nullptr;
        this -> truth_map_edge = nullptr; 

        this -> data_graph = nullptr; 
        this -> data_node = nullptr; 
        this -> data_edge = nullptr; 

        this -> truth_graph = nullptr; 
        this -> truth_node = nullptr; 
        this -> truth_edge = nullptr; 
    }

}; 


#endif

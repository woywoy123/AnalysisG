#include <switchboards.h>
#include <element.h>
#include <report.h>
#include <folds.h>

void graph_hdf5_w::flush_data(){
    free(this -> hash); 
    free(this -> filename); 
    free(this -> edge_index); 

    free(this -> data_map_graph); 
    free(this -> data_map_node); 
    free(this -> data_map_edge); 

    free(this -> truth_map_graph); 
    free(this -> truth_map_node);
    free(this -> truth_map_edge); 

    free(this -> data_graph); 
    free(this -> data_node); 
    free(this -> data_edge); 

    free(this -> truth_graph); 
    free(this -> truth_node); 
    free(this -> truth_edge); 

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

void graph_hdf5::export_gr(graph_hdf5_w* grw){
    grw -> num_nodes       = this -> num_nodes; 
    grw -> event_index     = this -> event_index;
    grw -> event_weight    = this -> event_weight; 
    
    grw -> hash            = const_cast<char*>(this -> hash.data());          
    grw -> filename        = const_cast<char*>(this -> filename.data());      
    grw -> edge_index      = const_cast<char*>(this -> edge_index.data());    
    
    grw -> data_map_graph  = const_cast<char*>(this -> data_map_graph.data()); 
    grw -> data_map_node   = const_cast<char*>(this -> data_map_node.data()); 
    grw -> data_map_edge   = const_cast<char*>(this -> data_map_edge.data());  
    
    grw -> truth_map_graph = const_cast<char*>(this -> truth_map_graph.data());
    grw -> truth_map_node  = const_cast<char*>(this -> truth_map_node.data());    
    grw -> truth_map_edge  = const_cast<char*>(this -> truth_map_edge.data());        
    
    grw -> data_graph      = const_cast<char*>(this -> data_graph.data());    
    grw -> data_node       = const_cast<char*>(this -> data_node.data());     
    grw -> data_edge       = const_cast<char*>(this -> data_edge.data());     
    
    grw -> truth_graph     = const_cast<char*>(this -> truth_graph.data());   
    grw -> truth_node      = const_cast<char*>(this -> truth_node.data());    
    grw -> truth_edge      = const_cast<char*>(this -> truth_edge.data());   
}

void graph_hdf5_w::import_gr(graph_hdf5* w){

    w -> num_nodes       = this -> num_nodes; 
    w -> event_index     = this -> event_index;
    w -> event_weight    = this -> event_weight; 
    
    w -> hash            = std::string(this -> hash);          
    w -> filename        = std::string(this -> filename);      
    w -> edge_index      = std::string(this -> edge_index);    
    
    w -> data_map_graph  = std::string(this -> data_map_graph); 
    w -> data_map_node   = std::string(this -> data_map_node); 
    w -> data_map_edge   = std::string(this -> data_map_edge);  
    
    w -> truth_map_graph = std::string(this -> truth_map_graph);
    w -> truth_map_node  = std::string(this -> truth_map_node);    
    w -> truth_map_edge  = std::string(this -> truth_map_edge);        
    
    w -> data_graph      = std::string(this -> data_graph);    
    w -> data_node       = std::string(this -> data_node);     
    w -> data_edge       = std::string(this -> data_edge);     
    
    w -> truth_graph     = std::string(this -> truth_graph);   
    w -> truth_node      = std::string(this -> truth_node);    
    w -> truth_edge      = std::string(this -> truth_edge); 
}




std::string model_report::print(){
    std::string msg = "Run Name: " + this -> run_name; 
    msg += " Epoch: " + std::to_string(this -> epoch); 
    msg += " K-Fold: " + std::to_string(this -> k+1); 
    msg += "\n"; 
    msg += "__________ LOSS FEATURES ___________ \n"; 
    msg += this -> prx(&this -> loss_graph, "Graph Loss");
    msg += this -> prx(&this -> loss_node, "Node Loss"); 
    msg += this -> prx(&this -> loss_edge, "Edge Loss"); 

    msg += "__________ ACCURACY FEATURES ___________ \n"; 
    msg += this -> prx(&this -> accuracy_graph, "Graph Accuracy");
    msg += this -> prx(&this -> accuracy_node, "Node Accuracy"); 
    msg += this -> prx(&this -> accuracy_edge, "Edge Accuracy"); 
    if (this -> current_lr.size()){msg += "Current LR: ";}
    for (size_t x(0); x < this -> current_lr.size(); ++x){msg += std::to_string(this -> current_lr[x]) + " | ";}
    return msg; 
}

std::string model_report::prx(std::map<mode_enum, std::map<std::string, float>>* data, std::string title){
    bool trig = false; 
    std::string out = ""; 
    std::map<std::string, float>::iterator itf; 
    std::map<mode_enum, std::map<std::string, float>>::iterator itx; 
    for (itx = data -> begin(); itx != data -> end(); ++itx){
        if (!itx -> second.size()){return "";}
        if (!trig){out += title + ": \n"; trig = true;}
        out += model_mode(itx -> first) + " -> ";  
        
        for (itf = itx -> second.begin(); itf != itx -> second.end(); ++itf){
            out += itf -> first + ": " + std::to_string(itf -> second) + " | "; 
        }
        out += "\n"; 
    }
    return out; 
}



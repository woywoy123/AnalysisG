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
        switch (itx -> first){
            case mode_enum::training:   out += "Training -> "; break;
            case mode_enum::validation: out += "Validation -> "; break;
            case mode_enum::evaluation: out += "Evaluation -> "; break; 
        }
        for (itf = itx -> second.begin(); itf != itx -> second.end(); ++itf){
            out += itf -> first + ": " + std::to_string(itf -> second) + " | "; 
        }
        out += "\n"; 
    }
    return out; 
}



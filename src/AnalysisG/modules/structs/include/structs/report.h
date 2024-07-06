#ifndef REPORT_STRUCTS_H
#define REPORT_STRUCTS_H
#include <string>
#include <map>

enum mode_enum {training, validation, evaluation}; 

struct model_report {
    int k;
    int epoch; 
    bool is_complete = false; 

    std::map<mode_enum, std::map<std::string, float>> loss_graph = {}; 
    std::map<mode_enum, std::map<std::string, float>> loss_node = {}; 
    std::map<mode_enum, std::map<std::string, float>> loss_edge = {}; 

    std::map<mode_enum, std::map<std::string, float>> accuracy_graph = {}; 
    std::map<mode_enum, std::map<std::string, float>> accuracy_node = {}; 
    std::map<mode_enum, std::map<std::string, float>> accuracy_edge = {}; 

    std::string run_name;
    std::string mode;
    
    float progress;

    std::string print(){
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

    std::string prx(
            std::map<mode_enum, std::map<std::string, float>>* data, 
            std::string title
    ){
        bool trig = false; 
        std::string out = ""; 
        std::map<std::string, float>::iterator itf; 
        std::map<mode_enum, std::map<std::string, float>>::iterator itx = data -> begin(); 
        for (; itx != data -> end(); ++itx){
            if (!itx -> second.size()){return "";}
            if (!trig){out += title + ": \n"; trig = true;}
            itf = itx -> second.begin(); 
            switch (itx -> first){
                case mode_enum::training:   out += "Training -> "; break;
                case mode_enum::validation: out += "Validation -> "; break;
                case mode_enum::evaluation: out += "Evaluation -> "; break; 
            }

            for (; itf != itx -> second.end(); ++itf){
                out += itf -> first + ": "; 
                out += std::to_string(itf -> second) + " | "; 
            }
            out += "\n"; 
        }
        return out; 
    }
}; 

#endif

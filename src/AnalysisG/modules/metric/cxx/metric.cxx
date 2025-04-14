#include <templates/metric_template.h>

metric_t::~metric_t(){
    std::map<graph_enum, std::vector<variable_t*>>::iterator itv = this -> handl -> begin(); 
    for (; itv != this -> handl -> end(); ++itv){
        for (size_t x(0); x < itv -> second.size(); ++x){
            if (!itv -> second[x]){continue;}
            itv -> second[x] -> clear = true; 
            delete itv -> second[x]; 
            itv -> second[x] = nullptr; 
        }
    }
}

void metric_t::build(){
    std::map<graph_enum, std::vector<std::string>>::iterator it;
    for (it = this -> vars -> begin(); it != this -> vars -> end(); ++it){
        for (size_t x(0); x < it -> second.size(); ++x){ 
            if (!(*this -> handl)[it -> first][x]){continue;}
            this -> v_maps[it -> first][it -> second[x]] = x;  
            this -> h_maps[it -> first][it -> second[x]] = true;  
        }
    }
}

std::string metric_t::mode(){
    switch (this -> train_mode){
        case mode_enum::training:   return "training"; 
        case mode_enum::validation: return "validation"; 
        case mode_enum::evaluation: return "evaluation"; 
        default: return "undef"; 
    }
}



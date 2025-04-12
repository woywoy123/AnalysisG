#include <templates/metric_template.h>

void metric_template::set_name(std::string* name, metric_template* ev){
    ev -> _name = *name; 
    ev -> prefix = *name;
}

void metric_template::get_name(std::string* name, metric_template* ev){
    *name = ev -> _name; 
}

void metric_template::set_run_name(std::map<std::string, std::string>* rn_name, metric_template* ev){
    std::string msgn = "Invalid Syntax for RunNames. Expected: <ModelName>::epoch-<X>::k-<X>.pt"; 
    std::string pthn = "Invalid Path. Cannot find the model state: "; 
    std::map<std::string, std::string>::iterator itx = rn_name -> begin(); 
    for (; itx != rn_name -> end(); ++itx){
        if (ev -> _run_names.count(itx -> first)){continue;}
        std::vector<std::string> names = ev -> split(itx -> first, "::");
        if (!ev -> is_file(itx -> second)){ev -> failure(pthn + itx -> second); return;}

        if (names.size() != 3){ev -> failure(msgn); return;}
        if (!ev -> has_string(&names[1], "epoch-")){ev -> failure(msgn); return;}
        if (!ev -> has_string(&names[2], "k-")){ev -> failure(msgn); return;}

        std::string mdl = names[0]; 
        std::string epc = ev -> split(names[1], "epoch-")[1];
        std::string kpc = ev -> split(names[2], "k-")[1]; 
        ev -> success("Adding: " + mdl + " @ Epoch " + epc + " using K-" + kpc); 
        ev -> _run_names[itx -> first] = itx -> second;
        ev -> _epoch_kfold[mdl][std::stoi(epc)][std::stoi(kpc)-1] = itx -> second;
    }
}

void metric_template::get_run_name(std::map<std::string, std::string>* rn_name, metric_template* ev){ 
    *rn_name = ev -> _run_names; 
}

void metric_template::set_variables(std::map<std::string, std::string>* rn_name, metric_template* ev){
    std::string msgn = "Invalid Syntax for Variables. Expected: <ModelName>::<Level>(data, truth, prediction)::<Type(edge, node, graph, extra)>::<variable>(index, pt, njets, ...)"; 
    std::map<std::string, std::string>::iterator itx = rn_name -> begin(); 
    for (; itx != rn_name -> end(); ++itx){
        if (ev -> _variables.count(itx -> first)){continue;}
        std::vector<std::string> varK = ev -> split(itx -> first, "::");
        if (varK.size() != 4){ev -> failure(msgn); return;}
        std::string mdl = varK[0]; 
        std::string var = varK[3]; 

        bool has_t = varK[1] == "truth"; 
        bool has_d = varK[1] == "data"; 
        bool has_p = varK[1] == "prediction"; 

        bool is_g  = varK[2] == "graph"; 
        bool is_n  = varK[2] == "node"; 
        bool is_e  = varK[2] == "edge"; 
        bool is_p  = varK[2] == "extra"; 

        graph_enum type; 
        if      (has_d && is_e && var == "index"){type = graph_enum::edge_index;}
        else if (has_d && is_n && var == "index"){type = graph_enum::batch_index;}
        else if (has_d && is_g && var == "index"){type = graph_enum::batch_events;}
        else if (has_d && is_g && var == "weight"){type = graph_enum::weight;}
        else if (has_t && is_g){type = graph_enum::truth_graph;}
        else if (has_t && is_n){type = graph_enum::truth_node;}
        else if (has_t && is_e){type = graph_enum::truth_edge;}
        else if (has_d && is_g){type = graph_enum::data_graph;}
        else if (has_d && is_n){type = graph_enum::data_node;}
        else if (has_d && is_e){type = graph_enum::data_edge;}
        else if (has_p && is_g){type = graph_enum::pred_graph;}
        else if (has_p && is_n){type = graph_enum::pred_node;}
        else if (has_p && is_e){type = graph_enum::pred_edge;}
        else if (has_p && is_p){type = graph_enum::pred_extra;}
        else {ev -> failure(msgn); continue;}
        ev -> _var_type[mdl][type].push_back(var); 
        ev -> _variables[itx -> first] = itx -> second;
    }
}

void metric_template::get_variables(std::map<std::string, std::string>* rn_name, metric_template* ev){ 
    *rn_name = ev -> _variables;
}



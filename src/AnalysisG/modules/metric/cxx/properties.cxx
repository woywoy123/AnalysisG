#include <templates/metric_template.h>

void metric_template::set_name(std::string* name, metric_template* ev){
    ev -> _name = *name; 
    ev -> prefix = *name;
}

void metric_template::get_name(std::string* name, metric_template* ev){
    *name = ev -> _name; 
}

void metric_template::set_run_name(std::map<std::string, std::string>* rn_name, metric_template* ev){
    std::string msgn = "Invalid Syntax for RunNames. Expected: <ModelName>::epoch-<X>::k-<X>"; 
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
    std::string msgn = "Invalid Syntax for Variables. Expected: <ModelName>::<Type(edge, node, graph)>::<model-var>"; 
    std::string keyn = "Invalid Syntax for Variables. Expected: <Level(data, truth)>::<out>"; 

    std::map<std::string, std::string>::iterator itx = rn_name -> begin(); 
    for (; itx != rn_name -> end(); ++itx){
        if (ev -> _variables.count(itx -> first)){continue;}
        std::vector<std::string> var = ev -> split(itx -> first, "::");
        if (var.size() != 3){ev -> failure(msgn); return;}
        std::string mdl = var[0]; 

        graph_enum type; 
        bool ist = false; 
        if (ev -> has_string(&itx -> second, "truth::")){ist = true;}
        else if (ev -> has_string(&itx -> second, "data::")){ist = false;}
        else {ev -> failure(keyn); return;}

        if (ev -> has_string(&var[1], "edge")){type = (!ist) ? graph_enum::data_edge : graph_enum::truth_edge;}
        else if (ev -> has_string(&var[1], "node")){type = (!ist) ? graph_enum::data_node : graph_enum::truth_node;}
        else if (ev -> has_string(&var[1], "graph")){type = (!ist) ? graph_enum::data_graph : graph_enum::truth_graph;}
        else {ev -> failure(msgn); return;}

        ev -> _var_type[mdl][type].push_back(var[2]); 
        ev -> _variables[itx -> first] = itx -> second;
    }
}

void metric_template::get_variables(std::map<std::string, std::string>* rn_name, metric_template* ev){ 
    *rn_name = ev -> _variables;
}



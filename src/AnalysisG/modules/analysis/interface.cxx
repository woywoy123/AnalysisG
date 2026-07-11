#include <AnalysisG/analysis.h>

void analysis::add_samples(std::string path, std::string label){
    if (this -> ends_with(&path, ".root")){
        this -> file_labels[this -> absolute_path(path)] = label;
        return; 
    }
    if (this -> ends_with(&path, "*")){
        std::vector<std::string> vx = this -> ls(path);
        for (size_t x(0); x < vx.size(); ++x){this -> add_samples(vx[x], label);}
        return; 
    }
}

void analysis::add_selection_template(selection_template* sel){this -> selection_names[sel -> name] = sel;}
void analysis::add_event_template(event_template* ev, std::string label){this -> event_labels[label] = ev;}
void analysis::add_graph_template(graph_template* ev, std::string label){this -> graph_labels[label][ev -> name] = ev;}

void analysis::add_metric_template(metric_template* mx, model_template* mdl){
    std::string name_m = std::string(mx -> name) + "/" + std::string(mdl -> name); 
    bool dup = this -> metric_names.count(name_m); 
    if (dup){this -> warning("Duplicate input"); return;}
    metric_template* cl = mx -> clone(1); 
    model_template* md  = mdl -> clone(1);
    cl -> link(md); 
    this -> metric_names[name_m]  = cl; 
    this -> model_metrics[name_m] = md; 
}

void analysis::add_model(model_template* model, optimizer_params_t* op, std::string run_name){
    std::tuple<model_template*, optimizer_params_t*> para = {model, op}; 
    this -> model_session_names.push_back(run_name); 
    this -> model_sessions.push_back(para);  
}

void analysis::add_model(model_template* model, std::string run_name){
    this -> model_session_names.push_back(run_name); 
    this -> model_inference[run_name] = model; 
}



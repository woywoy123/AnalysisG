#include <generators/analysis.h>

analysis::analysis(){
    this -> prefix = "Analysis"; 
    this -> tracer  = new sampletracer(); 
    this -> loader  = new dataloader(); 
    this -> reader  = new io(); 
}

analysis::~analysis(){
    std::map<std::string, optimizer*>::iterator itt = this -> trainer.begin();
    for (; itt != this -> trainer.end(); ++itt){delete itt -> second;}
    delete this -> loader; 
    delete this -> tracer; 
    delete this -> reader; 
}

void analysis::add_samples(std::string path, std::string label){
    this -> file_labels[path] = label;   
}

void analysis::add_event_template(event_template* ev, std::string label){
    this -> event_labels[label] = ev; 
}

void analysis::add_graph_template(graph_template* ev, std::string label){
    this -> graph_labels[label][ev -> name] = ev; 
}

void analysis::add_selection_template(selection_template* sel){
    this -> selection_names[sel -> name] = sel; 
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

void analysis::build_project(){
    this -> create_path(this -> m_settings.output_path); 
    std::string model_path = this -> m_settings.output_path; 

    for (size_t x(0); x < this -> model_session_names.size(); ++x){
        model_template* mdl = std::get<0>(this -> model_sessions.at(x)); 
        std::string pth = model_path + "/"; 
        pth += std::string(mdl -> name) + "/"; 
        pth += this -> model_session_names[x] + "/"; 
        mdl-> model_checkpoint_path = pth; 
    }
}

void analysis::start(){
    this -> success("+============================+"); 
    this -> success("| Starting Analysis Session! |");
    this -> success("+============================+"); 

    bool trig = false; 
    int threads_ = this -> m_settings.threads; 
    std::string path_data = this -> m_settings.training_dataset; 
    std::string pth_cache = this -> m_settings.graph_cache; 
    std::vector<std::string> cache = {}; 
    if (pth_cache.size()){cache = this -> ls(pth_cache, ".h5");}
    for (size_t x(0); x < cache.size(); ++x){
        if (!this -> has_string(&cache[x], "0x")){continue;}
        trig = true; 
        break; 
    }

    if (!trig){
        this -> build_events(); 
        if (this -> selection_names.size()){this -> build_selections();}
        if (this -> graph_labels.size()){this -> build_graphs();}
        this -> tracer -> compile_objects(threads_); 
    }

    if (this -> selection_names.size()){
        this -> tracer -> fill_selections(&this -> selection_names);
    } 

    if (pth_cache.size() && !trig){
        this -> build_dataloader(false); 
        this -> loader -> dump_graphs(pth_cache, threads_);
        this -> info("Validating the graph cache..."); 
        std::string msg1 = "Graph cache has been validated!"; 
        std::string msg2 = "Failed to validate the graph cache! Skipping..."; 
        if (this -> loader -> restore_graphs(pth_cache, threads_)){this -> success(msg1);}
        else {this -> failure(msg2);}
    }
    else if (trig){this -> loader -> restore_graphs(pth_cache, threads_);}

    if (!this -> loader -> restore_dataset(path_data) && path_data.size()){
        this -> build_dataloader(true);
    }

    if (this -> model_sessions.size()){
        this -> build_dataloader(true); 
        this -> build_project(); 
        this -> build_model_session();  
    }

    if (this -> model_inference.size()){
        this -> build_dataloader(false);
        this -> build_inference(); 
    }
}

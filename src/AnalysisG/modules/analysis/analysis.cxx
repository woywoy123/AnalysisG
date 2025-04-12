#include <ROOT/RDataFrame.hxx>
#include <AnalysisG/analysis.h>
#include <AnalysisG/cfg.h>
#include <TSystem.h>
#include <thread>

analysis::analysis(){
    this -> prefix = "Analysis"; 
    this -> tags    = new std::vector<folds_t>(); 
    this -> tracer  = new sampletracer(); 
    this -> loader  = new dataloader(); 
    this -> reader  = new io(); 
    std::string cur = this -> absolute_path("./");

    std::string tmp = std::string(dict_path) + "pcm/";
    this -> create_path(tmp);
    int opx = static_cast<int>(data_enum::undef);
    int opc = this -> ls(tmp, ".pcm").size(); 
    if (opx-6 > opc){this -> info("Building PCM files... to:" + tmp);}
    gSystem -> SetBuildDir(tmp.c_str(), true); 
    gSystem -> ChangeDirectory(tmp.c_str()); 
    gSystem -> AddDynamicPath(tmp.c_str());
    gSystem -> SetAclicMode(TSystem::kOpt); 

    std::string mta = std::string(dict_path) + "structs/include/structs/meta.h"; 
    std::thread* tm = nullptr; 
    tm = new std::thread(buildDict, "meta_t"   , mta); 
    tm -> join(); delete tm; 
    tm = new std::thread(buildDict, "weights_t", mta); 
    tm -> join(); delete tm; 
    tm = new std::thread(buildAll); 
    tm -> join(); delete tm; 
    gSystem -> ChangeDirectory(cur.c_str()); 
}

analysis::~analysis(){
    for (size_t x(0); x < this -> tags -> size(); ++x){(*this -> tags)[x].flush_data();}
    delete this -> tags; 

    flush(&this -> trainer);
    flush(&this -> model_metrics);
    flush(&this -> metric_names); 
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

void analysis::add_metric_template(metric_template* mx, model_template* mdl){
    this -> safe_clone(&this -> model_metrics, mdl); this -> safe_clone(&this -> metric_names, mx);  
    if (this -> metric_names[mx -> name] -> link(this -> model_metrics[mdl -> name])){return;}
    abort();
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

    if (!this -> model_metrics.size()){return;}






}

void analysis::check_cache(){
    std::string pth_cache = this -> m_settings.graph_cache; 
    std::vector<std::string> cache = this -> ls(pth_cache, ".h5");

    std::map<std::string, std::string> relabel = {}; 
    std::map<std::string, std::string>::iterator tx = this -> file_labels.begin(); 
    for (; tx != this -> file_labels.end(); ++tx){
        std::vector<std::string> graph_cache = {}; 
        if (this -> graph_labels.count(tx -> second)){
            std::map<std::string, graph_template*>::iterator itg; 
            itg = this -> graph_labels[tx -> second].begin();
            for (; itg != this -> graph_labels[tx -> second].end(); ++itg){
                graph_cache.push_back(itg -> first);
                this -> graph_types[itg -> first]; 
            }
        }
        std::vector<std::string> files = this -> ls(tx -> first, ".root"); 
        if (!files.size()){files = {tx -> first};}
        for (size_t x(0); x < files.size(); ++x){
            std::string file_n = files[x]; 
            std::vector<std::string> spl = this -> split(file_n, "/"); 
            std::string fname = this -> hash(spl[spl.size()-1]) + "-" + spl[spl.size()-1]; 
            this -> replace(&fname, ".root", ".h5"); 
            int s = 0; 
            for (size_t y(0); y < graph_cache.size(); ++y){
                this -> in_cache[file_n][graph_cache[y] + "/" + fname] = false;
                ++s; 
            }

            int sg = 0; 
            for (size_t y(0); y < cache.size(); ++y){
                std::vector<std::string> spl_ = this -> split(cache[y], "/"); 
                std::string fname_ = spl_[spl_.size()-2] + "/" + spl_[spl_.size()-1]; 
                if (this -> has_string(&cache[y], "/." + spl_[spl_.size()-1])){continue;}
                if (!this -> in_cache[file_n].count(fname_)){continue;}
                this -> in_cache[file_n][fname_] = true;
                ++sg; 
            }

            if (s == sg && s){this -> skip_event_build[file_n] = true;}
            else {this -> skip_event_build[file_n] = false;}
            relabel[file_n] = tx -> second; 
        }
    }
    this -> file_labels = relabel; 
}

void analysis::fetchtags(){
    if (this -> tags -> size()){return;}
    io* io_g = new io(); 
    io_g -> start(this -> m_settings.training_dataset, "read"); 
    io_g -> read(this -> tags, "kfolds"); 
    io_g -> end(); 
    delete io_g; 
}

void analysis::start(){
    this -> tracer -> shush = this -> m_settings.debug_mode; 
    if (this -> tracer -> shush){this -> m_settings.threads = 1;}
    if (this -> m_settings.pretagevents){this -> fetchtags();}
    this -> create_path(this -> m_settings.output_path + "/"); 

    if (!this -> started){
        this -> success("+============================+"); 
        this -> success("| Starting Analysis Session! |");
        this -> success("+============================+"); 
        this -> check_cache();
        this -> build_events(); 
    }

    if (!this -> started){
        if (!this -> event_labels.size() && !this -> graph_labels.size()){
            this -> reader -> import_settings(&this -> m_settings); 
            std::map<std::string, std::string>::iterator itf = this -> file_labels.begin(); 
            for (; itf != this -> file_labels.end(); ++itf){this -> reader -> root_files[itf -> first] = true;}
            this -> reader -> check_root_file_paths();
            this -> reader -> scan_keys(); 
            this -> meta_data = this -> reader -> meta_data; 
        }
        this -> started = true; 
        return; 
    }

    int threads_ = this -> m_settings.threads; 
    ROOT::EnableImplicitMT(threads_); 

    std::string pth_cache = this -> m_settings.graph_cache; 
    this -> loader -> setting = &this -> m_settings; 

    if (pth_cache.size() && !this -> ends_with(&pth_cache, "/")){pth_cache += "/";}
    if (this -> selection_names.size()){this -> build_selections();}
    if (this -> graph_labels.size()){this -> build_graphs();}
    this -> tracer -> compile_objects(threads_); 
    if (this -> selection_names.size()){return this -> tracer -> fill_selections(&this -> selection_names);} 
    this -> build_dataloader(false); 
    this -> build_metric_folds();

    if (this -> m_settings.build_cache && !this -> loader -> data_set -> size()){return;}
    if (pth_cache.size() && this -> loader -> data_set -> size()){this -> loader -> dump_graphs(pth_cache, threads_);}
    else if (pth_cache.size() && this -> file_labels.size()){
        std::vector<std::string> cached = {}; 
        std::map<std::string, std::string>::iterator itg = this -> graph_types.begin(); 
        for (; itg != this -> graph_types.end(); ++itg){
            std::map<std::string, std::string>::iterator itc = this -> file_labels.begin(); 
            for (; itc != this -> file_labels.end(); ++itc){
                std::vector<std::string> spl = this -> split(itc -> first, "/");
                std::string fname = this -> hash(spl[spl.size()-1]) + "-" + spl[spl.size()-1]; 
                this -> replace(&fname, ".root", ".h5"); 
                cached.push_back(pth_cache + itg -> first + "/" + fname);  
            }
        }
        this -> loader -> restore_graphs(cached, threads_); 
    }
    else if (pth_cache.size()){this -> loader -> restore_graphs(pth_cache, threads_);}
    if (!this -> build_metric()){return;}

    if (this -> model_sessions.size()){
        if (!this -> loader -> data_set -> size()){return this -> failure("No Dataset was found for training. Aborting...");}
        this -> build_dataloader(true); 
        this -> loader -> start_cuda_server(); 
        this -> build_project(); 
        this -> build_model_session();  
    }

    if (this -> model_inference.size()){
        if (!this -> loader -> data_set -> size()){return this -> failure("No Dataset was found for inference. Aborting...");}
        this -> loader -> start_cuda_server(); 
        this -> build_inference(); 
    }
}

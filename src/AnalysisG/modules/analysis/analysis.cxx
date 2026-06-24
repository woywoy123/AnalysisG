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
    tm = new std::thread(buildDict, "meta_t"   , mta); tm -> join(); delete tm; 
    tm = new std::thread(buildDict, "weights_t", mta); tm -> join(); delete tm; 
    tm = new std::thread(buildAll); tm -> join(); delete tm; 
    gSystem -> ChangeDirectory(cur.c_str()); 
}

analysis::~analysis(){
    for (size_t x(0); x < this -> tags -> size(); ++x){(*this -> tags)[x].flush_data();}
//    this -> pflush(&this -> tags); 
//    this -> mflush(&this -> trainer);
//    this -> mflush(&this -> model_metrics);
//    this -> mflush(&this -> metric_names); 
//    this -> pflush(&this -> tracer); 
//    this -> pflush(&this -> reader); 
//    this -> pflush(&this -> loader); 
}

void analysis::add_samples(std::string path, std::string label){
    if (this -> ends_with(&path, ".root")){path = this -> absolute_path(path);}
    else if (this -> ends_with(&path, "*")){
        std::vector<std::string> vx = this -> ls(path);
        for (size_t x(0); x < vx.size(); ++x){this -> add_samples(vx[x], label);}
        return; 
    }
    else {return;}
    if (!this -> ends_with(&path, ".root")){return;}
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
    std::map<std::string, metric_template*>::iterator itm = this -> metric_names.begin();
    for (; itm != this -> metric_names.end(); ++itm){
        itm -> second -> outdir = model_path + "/metrics/" + itm -> first;
    }
}

void analysis::check_cache(){
    std::string pth_cache = this -> m_settings.graph_cache; 
    std::vector<std::string> cache = this -> ls(pth_cache, ".h5");

    std::map<std::string, std::string> relabel = {}; 
    std::map<std::string, std::string>::iterator tx = this -> file_labels.begin(); 

    for (; tx != this -> file_labels.end(); ++tx){
        std::string base = tx -> first; 
        std::string lbl  = tx -> second; 
        std::vector<std::string> graph_cache = {}; 
        if (this -> graph_labels.count(lbl)){

            std::map<std::string, graph_template*>::iterator itg; 
            itg = this -> graph_labels[lbl].begin();
            for (; itg != this -> graph_labels[lbl].end(); ++itg){

                graph_cache.push_back(itg -> first);
                this -> graph_types[itg -> first]; 
            
            }
        }
        std::vector<std::string> files = this -> ls(lbl, ".root"); 
        if (this -> ends_with(&base, ".root")){files.push_back(tx -> first);}
        for (size_t x(0); x < files.size(); ++x){

            std::string file_n = files[x]; 
            std::string fname = this -> get_splits(&file_n, "/"); 
            fname = this -> hash(fname) + "-" + fname;  
            this -> replace(&fname, ".root", ".h5"); 
            int s = 0; 
            for (size_t y(0); y < graph_cache.size(); ++y){
                this -> in_cache[file_n][graph_cache[y] + "/" + fname] = false;
                ++s; 
            }

            int sg = 0; 
            for (size_t y(0); y < cache.size(); ++y){
                std::string fname_ = this -> get_splits(&cache[y], "/", -2) + "/"; 
                std::string fnameK = this -> get_splits(&cache[y], "/", -1); 
                fname_ = fname_ + fnameK; 

                if (this -> has_string(&cache[y], "/." + fnameK)){continue;}
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

    int threads_ = this -> m_settings.threads; 
    int intra_th = this -> m_settings.intra_th; 

    std::string pth_cache = this -> m_settings.graph_cache; 
    bool build_gr_cache   = this -> m_settings.build_cache; 
    bool load_gr_cache    = pth_cache.size() > 0; 

    std::map<std::string, bool>* root_f = &this -> reader -> root_files; 


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
            for (; itf != this -> file_labels.end(); ++itf){(*root_f)[itf -> first] = true;}
            this -> reader -> check_root_file_paths();
            this -> reader -> scan_keys(); 
            this -> meta_data = this -> reader -> meta_data; 
        }
        this -> started = true; 
        return; 
    }

    ROOT::EnableImplicitMT(threads_); 
    this -> loader -> setting = &this -> m_settings; 

    if (load_gr_cache && !this -> ends_with(&pth_cache, "/")){pth_cache += "/";}
    if (this -> selection_names.size()){this -> build_selections();}
    if (this -> graph_labels.size()){this -> build_graphs();}

    this -> tracer -> compile_objects(threads_, intra_th); 
  
    if (this -> selection_names.size()){
        return this -> tracer -> fill_selections(&this -> selection_names);
    } 
    this -> build_dataloader(false); 
    this -> build_metric_folds();

    if (load_gr_cache  && this -> dsize()){this -> loader -> dump_graphs(pth_cache, threads_);}
    if (build_gr_cache && this -> dsize()){this -> success("Graph Caches Build: " + pth_cache);}
    else if (load_gr_cache && this -> file_labels.size()){
        std::vector<std::string> cached = {}; 
        std::map<std::string, std::string>::iterator itg = this -> graph_types.begin(); 
        for (; itg != this -> graph_types.end(); ++itg){
            std::map<std::string, std::string>::iterator itc = this -> file_labels.begin(); 
            for (; itc != this -> file_labels.end(); ++itc){

                std::string fname = itc -> first; 
                fname = this -> get_splits(&fname, "/"); 
                fname = this -> hash(fname) + "-" + fname; 
                this -> replace(&fname, ".root", ".h5"); 

                cached.push_back(pth_cache + itg -> first + "/" + fname);  
            }
        }
        this -> loader -> restore_graphs(cached, threads_); 
    }
    else if (load_gr_cache){this -> loader -> restore_graphs(pth_cache, threads_);}
    
    if (!this -> build_metric()){return;}
    if (this -> model_sessions.size()){
        if (!this -> dsize()){return this -> failure("No Dataset was found from training. Aborting...");}
        this -> loader -> restore_dataset(this -> m_settings.training_dataset); 
        this -> build_dataloader(true); 
        this -> loader -> start_cuda_server(); 
        this -> build_project(); 
        this -> build_model_session();  
    }

    if (this -> model_inference.size()){
        if (!this -> dsize()){
            return this -> failure("No Dataset was found for inference. Aborting...");
        }
        this -> loader -> start_cuda_server(); 
        this -> build_inference(); 
    }
}

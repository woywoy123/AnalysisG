#include <generators/analysis.h>

void analysis::build_model_session(){
    if (!this -> model_sessions.size()){return this -> info("No Models Specified. Skipping.");}
    std::vector<int> kfold = this -> m_settings.kfold; 
    if (!kfold.size()){for (int k(0); k < this -> m_settings.kfolds; ++k){kfold.push_back(k);}}
    else {for (int k(0); k < kfold.size(); ++k){kfold[k] = kfold[k]-1;}}
    this -> m_settings.kfold = kfold; 
    int th_ = this -> m_settings.threads; 
    
    for (size_t x(0); x < this -> model_sessions.size(); ++x){
        std::string name = this -> model_session_names[x]; 

        optimizer* optim = new optimizer();
        optim -> m_settings = this -> m_settings; 
        optim -> m_settings.run_name = name; 
        optim -> import_dataloader(this -> loader);

        std::tuple<model_template*, optimizer_params_t*>* para = nullptr; 
        para = &this -> model_sessions[x]; 
        
        this -> info("Transferring Graphs to device."); 
        this -> loader -> datatransfer(std::get<0>(*para) -> m_option, th_); 
        this -> success("Transfer Complete!"); 

        optim -> import_model_sessions(para); 
        optim -> check_model_sessions(this -> m_settings.num_examples, &this -> reports); 
        this -> trainer[name] = optim; 
    }

    auto lamb = [](optimizer* op, int k){op -> launch_model(k);}; 

    this -> threads = {}; 
    std::map<std::string, optimizer*>::iterator itr = this -> trainer.begin(); 
    for (; itr != this -> trainer.end(); ++itr){
        for (int k(0); k < this -> m_settings.kfold.size(); ++k){
            int k_ = this -> m_settings.kfold[k]; 
            std::vector<graph_t*>* check = this -> loader -> get_k_train_set(k_); 
            if (!check){continue;}
 
            if (this -> m_settings.debug_mode){itr -> second -> launch_model(k_);}
            else {this -> threads.push_back(new std::thread(lamb, itr -> second, k_));}
        }
    }
}

std::map<std::string, float> analysis::progress(){
    std::map<std::string, float> output; 
    std::map<std::string, model_report*>::iterator itx;
    for (itx = this -> reports.begin(); itx != this -> reports.end(); ++itx){
        output[itx -> first] = itx -> second -> progress*100; 
    }
    return output; 
}

std::map<std::string, std::string> analysis::progress_mode(){
    std::map<std::string, std::string> output; 
    std::map<std::string, model_report*>::iterator itx;
    for (itx = this -> reports.begin(); itx != this -> reports.end(); ++itx){
        std::string o = itx -> second -> mode;
        o += "| k-" + std::to_string(itx -> second -> k+1); 
        o += "| RunName: " + itx -> second -> run_name; 
        o += "| Epoch: " + std::to_string(itx -> second -> epoch);
        output[itx -> first] = o; 
    }
    return output; 
}

std::map<std::string, std::string> analysis::progress_report(){
    std::map<std::string, std::string> output; 
    std::map<std::string, model_report*>::iterator itx;
    for (itx = this -> reports.begin(); itx != this -> reports.end(); ++itx){
        output[itx -> first] = itx -> second -> print(); 
        metrics* plt = itx -> second -> waiting_plot; 
        if (!plt){continue;}
        plt -> dump_plots(itx -> second -> k); 
        itx -> second -> waiting_plot = nullptr;  
    }
    this -> loader -> start_cuda_server(); 
    return output; 
}

std::map<std::string, bool> analysis::is_complete(){
    std::map<std::string, bool> output; 
    std::map<std::string, model_report*>::iterator itx = this -> reports.begin();
    for (; itx != this -> reports.end(); ++itx){output[itx -> first] = itx -> second -> is_complete;}
    return output; 
}

void analysis::attach_threads(){
    for (int x(0); x < this -> threads.size(); ++x){
        if (!this -> threads[x]){continue;}
        this -> threads[x] -> join(); 
        delete this -> threads[x]; 
        this -> threads[x] = nullptr; 
    }
}

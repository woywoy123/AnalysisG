#include <AnalysisG/analysis.h>
#include <ROOT/RDataFrame.hxx>

void analysis::build_model_session(){

    if (!this -> model_sessions.size()){return this -> info("No Models Specified. Skipping.");}
    std::vector<int> kfold = this -> m_settings.kfold; 
    if (!kfold.size()){for (int k(0); k < this -> m_settings.kfolds; ++k){kfold.push_back(k);}}
    else {for (size_t k(0); k < kfold.size(); ++k){kfold[k] = kfold[k]-1;}}
    this -> m_settings.kfold = kfold; 

    // --------------- transferring data graphs -------------- //
    std::map<int, torch::TensorOptions*> dev_map; 
    for (size_t x(0); x < this -> model_sessions.size(); ++x){
        torch::TensorOptions* op = std::get<0>(this -> model_sessions[x]) -> m_option; 
        int dx = op -> device().index();
        if (dev_map.count(dx)){continue;}
        dev_map[dx] = op; 
    }
    this -> loader -> datatransfer(&dev_map); 
    // --------------- transferring data graphs -------------- //

    for (size_t x(0); x < this -> model_sessions.size(); ++x){
        std::string name = this -> model_session_names[x]; 

        optimizer* optim = new optimizer();
        optim -> m_settings = this -> m_settings; 
        optim -> m_settings.run_name = name; 
        optim -> import_dataloader(this -> loader);
        this -> trainer[name] = optim; 

        std::tuple<model_template*, optimizer_params_t*> para = this -> model_sessions[x]; 
        optim -> import_model_sessions(&para); 

        for (size_t k(0); k < this -> m_settings.kfold.size(); ++k){
            int k_ = this -> m_settings.kfold[k]; 
            std::vector<graph_t*>* check = this -> loader -> get_k_train_set(k_); 
            if (!check){continue;}
            model_report* mx = nullptr;  
            if (this -> m_settings.debug_mode){initialize_loop(optim, k_, std::get<0>(para), std::get<1>(para), &mx);}
            else {this -> threads.push_back(new std::thread(initialize_loop, optim, k_, std::get<0>(para), std::get<1>(para), &mx));}
            while (!mx){std::this_thread::sleep_for(std::chrono::microseconds(10));}
            this -> reports[mx -> run_name + std::to_string(mx -> k)] = mx; 
        }
    }
}

std::map<std::string, std::vector<float>> analysis::progress(){
    std::map<std::string, std::vector<float>> output; 
    std::map<std::string, model_report*>::iterator itx;
    for (itx = this -> reports.begin(); itx != this -> reports.end(); ++itx){
        model_report* mr = itx -> second; 
        if (!mr -> num_evnt){mr -> num_evnt = 1;}
        mr -> progress = float(mr -> iters) / float(mr -> num_evnt); 
        output[itx -> first] = {itx -> second -> progress*100, float(mr -> iters), float(mr -> num_evnt)}; 
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
    return output; 
}

std::map<std::string, bool> analysis::is_complete(){
    std::map<std::string, bool> output; 
    std::map<std::string, model_report*>::iterator itx = this -> reports.begin();
    for (; itx != this -> reports.end(); ++itx){output[itx -> first] = itx -> second -> is_complete;}
    return output; 
}

void analysis::attach_threads(){this -> monitor(&this -> threads);}

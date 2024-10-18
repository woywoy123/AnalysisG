#include <generators/analysis.h>
#include <ROOT/RDataFrame.hxx>

static void initialize_loop(
        optimizer* op, int k, model_template* model, 
        optimizer_params_t* config, model_report** rep
){
   
//    ROOT::EnableImplicitMT(); 
    model_settings_t settings; 
    model -> clone_settings(&settings); 
    model_template* mk = model -> clone(); 
    std::string pth = model -> model_checkpoint_path; 

    mk -> import_settings(&settings); 
    mk -> set_optimizer(config -> optimizer); 
    mk -> initialize(config); 

    mk -> epoch = 0; 
    mk -> kfold = k+1; 
    for (int ep(0); ep < op -> m_settings.epochs; ++ep){

         // check if the next epoch has a file i+2;
        std::string pth_ = pth + "state/epoch-" + std::to_string(ep+1) + "/";  
        pth_ += "kfold-" + std::to_string(k+1) + "_model.pt"; 

        if (op -> m_settings.continue_training && op -> is_file(pth_)){continue;}
        if (!op -> m_settings.continue_training){break;} 
        mk -> epoch = ep+1;
        mk -> restore_state(); 
        break; 
    }

    std::vector<graph_t*> rnd = op -> loader -> get_random(1); 
    mk -> shush = true; 
    mk -> check_features(rnd[0]);
    op -> kfold_sessions[k] = mk;
    model_report* mr = op -> metric -> register_model(mk, k); 
    op -> reports[mr -> run_name + std::to_string(mr -> k)] = mr; 
    (*rep) = mr; 
    op -> launch_model(k);
}

void analysis::build_model_session(){
    if (!this -> model_sessions.size()){return this -> info("No Models Specified. Skipping.");}
    std::vector<int> kfold = this -> m_settings.kfold; 
    if (!kfold.size()){for (int k(0); k < this -> m_settings.kfolds; ++k){kfold.push_back(k);}}
    else {for (int k(0); k < kfold.size(); ++k){kfold[k] = kfold[k]-1;}}
    this -> m_settings.kfold = kfold; 
    int th_ = this -> m_settings.threads; 

    //torch::set_num_threads(144);
    //torch::set_num_interop_threads(1); 
    for (size_t x(0); x < this -> model_sessions.size(); ++x){
        std::string name = this -> model_session_names[x]; 

        optimizer* optim = new optimizer();
        optim -> m_settings = this -> m_settings; 
        optim -> m_settings.run_name = name; 
        optim -> import_dataloader(this -> loader);

        std::tuple<model_template*, optimizer_params_t*>* para = &this -> model_sessions[x]; 
        this -> info("Transferring Graphs to device."); 
        this -> loader -> datatransfer(std::get<0>(*para) -> m_option, th_); 
        this -> success("Transfer Complete!"); 
        this -> trainer[name] = optim; 
        optim -> import_model_sessions(para); 

        for (int k(0); k < this -> m_settings.kfold.size(); ++k){
            int k_ = this -> m_settings.kfold[k]; 
            std::vector<graph_t*>* check = this -> loader -> get_k_train_set(k_); 
            if (!check){continue;}
            model_report* mx = nullptr;  
            if (this -> m_settings.debug_mode){initialize_loop(optim, k_, std::get<0>(*para), std::get<1>(*para), &mx);}
            else {this -> threads.push_back(new std::thread(initialize_loop, optim, k_, std::get<0>(*para), std::get<1>(*para), &mx));}
            while (!mx){std::this_thread::sleep_for(std::chrono::microseconds(10));}
            this -> reports[mx -> run_name + std::to_string(mx -> k)] = mx; 
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

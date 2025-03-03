#include <generators/analysis.h>
#include <ROOT/RDataFrame.hxx>

void initialize_loop(
        optimizer* op, int k, model_template* model, 
        optimizer_params_t* config, model_report** rep
){
    ROOT::EnableImplicitMT(); 
    model_settings_t settings; 
    model -> clone_settings(&settings); 

    #ifdef PYC_CUDA
    c10::cuda::set_device(model -> m_option -> get_device()); 
    #endif 
    
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
        mk -> epoch = ep;
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
    auto lamb = [](dataloader* ld, torch::TensorOptions* op, size_t* num_ev, size_t* prg){ld -> datatransfer(op, num_ev, prg);};

    if (!this -> model_sessions.size()){return this -> info("No Models Specified. Skipping.");}
    std::vector<int> kfold = this -> m_settings.kfold; 
    if (!kfold.size()){for (int k(0); k < this -> m_settings.kfolds; ++k){kfold.push_back(k);}}
    else {for (size_t k(0); k < kfold.size(); ++k){kfold[k] = kfold[k]-1;}}
    this -> m_settings.kfold = kfold; 

    // --------------- transferring data graphs -------------- //
    std::map<int, bool> dev_map; 
    for (size_t x(0); x < this -> model_sessions.size(); ++x){
        dev_map[std::get<0>(this -> model_sessions[x]) -> m_option -> device().index()] = false; 
    }
  
    size_t num_thr = 0;  
    std::vector<std::thread*> trans(dev_map.size(), nullptr); 
    std::vector<std::string*> titles(dev_map.size(), nullptr); 
    std::vector<size_t> num_events(dev_map.size(), 0); 
    std::vector<size_t> prg_events(dev_map.size(), 0);

    std::tuple<model_template*, optimizer_params_t*>* para;
    this -> info("Transferring Graphs to device."); 
    for (size_t x(0); x < this -> model_sessions.size(); ++x){
        para = &this -> model_sessions[x]; 
        torch::TensorOptions* op = std::get<0>(*para) -> m_option;
        int dev = op -> device().index();
        if (dev_map[dev]){continue;}
        dev_map[dev] = true; 
        trans[num_thr]  = new std::thread(lamb, this -> loader, op, &num_events[num_thr], &prg_events[num_thr]); 
        titles[num_thr] = new std::string("Device" + std::to_string(dev)); 
        ++num_thr; 
    }
    std::thread* thr = new std::thread(this -> progressbar3, &prg_events, &num_events, &titles); 
    this -> monitor(&trans); 
    this -> success("Transfer Complete!"); 
    thr -> join(); delete thr; thr = nullptr; 
    // --------------- transferring data graphs -------------- //

    for (size_t x(0); x < this -> model_sessions.size(); ++x){
        std::string name = this -> model_session_names[x]; 

        optimizer* optim = new optimizer();
        optim -> m_settings = this -> m_settings; 
        optim -> m_settings.run_name = name; 
        optim -> import_dataloader(this -> loader);

        para = &this -> model_sessions[x]; 
        this -> trainer[name] = optim; 
        optim -> import_model_sessions(para); 

        for (size_t k(0); k < this -> m_settings.kfold.size(); ++k){
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

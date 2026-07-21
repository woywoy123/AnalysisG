#include <AnalysisG/analysis.h>
#include <structs/switchboards.h>

void analysis::build_metric_folds(){
    if (!this -> model_metrics.size()){return;}
    std::vector<int> kfolds = {};
    std::map<std::string, metric_template*>::iterator itm = this -> metric_names.begin();
    for (; itm != this -> metric_names.end(); ++itm){
        std::vector<int> kf = itm -> second -> get_kfolds(); 
        this -> unique_key(&kf, &kfolds); 
    }
    this -> m_settings.kfold = kfolds;
}

bool analysis::build_metric(){
    auto lamb =[this](
            bool mode, mode_enum mt, metric_model_t* mx, 
            std::map<std::string, std::vector<graph_t*>*>* cx
    ) -> long {
        if (!mode){return 0;}
        std::string key = "<|" + mx -> run_name + "|device-" + std::to_string(mx -> device); 
        key += "|model-mode:" + model_mode(mt); 
        key += (mt == mode_enum::evaluation) ? "|>" : "|kfold-" + std::to_string(mx -> kfold) + "|>"; 
        if (cx -> count(key)){mx -> batches[mt] = (*cx)[key]; return 0;}
        std::vector<graph_t*>* smpl = nullptr; 
        switch(mt){
            case mode_enum::training:   smpl = this -> loader -> get_k_train_set(mx -> kfold);      break;
            case mode_enum::validation: smpl = this -> loader -> get_k_validation_set(mx -> kfold); break;
            case mode_enum::evaluation: smpl = this -> loader -> get_test_set();                    break;
            default: break;   
        }
        if (!smpl){return 0;}
        (*cx)[key] = this -> loader -> build_batch(smpl, mx -> model, nullptr); 
        mx -> batches[mt] = (*cx)[key];
        return long((*cx)[key] -> size()); 
    }; 

    if (!this -> model_metrics.size()){return true;}
    size_t threads_ = this -> m_settings.threads; 
    size_t lx       = this -> dsize(); 

    bool tr = this -> m_settings.training; 
    bool va = this -> m_settings.validation; 
    bool ev = this -> m_settings.evaluation; 
    bool debug_mode = this -> m_settings.debug_mode + !threads_;  

    std::string pth_cache = this -> m_settings.graph_cache; 
    std::map<mode_enum, std::string> spl_cache = model_mode(&this -> m_settings.splt_graph_cache); 

    if (spl_cache[mode_enum::evaluation].size() && ev){
        this -> warning("Adding evaluation samples " + pth_cache + " to collection."); 
        this -> warning("Make sure that the directory has no duplicated events to prevent double counting."); 
        this -> warning("Assuming Evaluation for all graphs within specified cache path."); 
        if (spl_cache[mode_enum::evaluation].size()){pth_cache = spl_cache[mode_enum::evaluation];}
        this -> loader -> restore_graphs(spl_cache[mode_enum::evaluation], threads_, true); 
        lx = this -> dsize(); 
    }
    if (!lx){
        this -> failure("No Dataset was found for metrics. Aborting...");
        return false; 
    }

    // ------------- Compute the device and kfold hash -------------- //
    std::map<std::string, std::vector<graph_t*>*> batch_cache = {}; 
    std::vector<metric_model_t*> que = {}; 

    long smpls = 0; 
    this -> build_dataloader(true);
    std::map<int, torch::TensorOptions*> dev_map; 
    std::map<std::string, metric_template*>::iterator itm = this -> metric_names.begin();
    for (; itm != this -> metric_names.end(); ++itm){
        metric_template* mt = itm -> second; 
        size_t id = dev_map.size(); 

        // ------------- Get the devices -------------- //
        this -> info("Building vector for metric: " + itm -> first + " with: " + std::to_string(mt -> data -> size())); 
        std::map<int, torch::TensorOptions*>::iterator itt; 
        std::map<int, torch::TensorOptions*> dev_ = mt -> get_devices(); 
        for (itt = dev_.begin(); itt != dev_.end(); ++itt){
            if (dev_map.count(itt -> first)){continue;}
            dev_map[itt -> first] = itt -> second;
        }
        if (dev_map.size() != id){this -> loader -> datatransfer(&dev_map);}
       
        // ------------- Get the Batches -------------- // 
        std::map<int, size_t>::iterator itk; 
        std::map<int, std::map<int, size_t>>::iterator ite; 
        std::map<std::string, std::map<int, std::map<int, size_t>>>::iterator itx;
        for (itx = mt -> _epoch_kfold.begin(); itx != mt -> _epoch_kfold.end(); ++itx){
            for (ite = itx -> second.begin(); ite != itx -> second.end(); ++ite){
                for (itk = ite -> second.begin(); itk != ite -> second.end(); ++itk){
                    metric_model_t* wrk = mt -> data -> at(itk -> second); 
                    if ( !wrk -> verify() ){wrk -> failure("ERROR"); continue;}
                    smpls += lamb(tr, mode_enum::training  , wrk, &batch_cache); 
                    smpls += lamb(va, mode_enum::validation, wrk, &batch_cache); 
                    smpls += lamb(ev, mode_enum::evaluation, wrk, &batch_cache); 
                    wrk -> metrx = mt; 
                    que.push_back(wrk); 
                }
            } 
        } 
    }
    std::string msg = "Total Number Events: "; 
    msg += std::to_string(smpls) + " of Jobs Assigned: "; 
    msg += std::to_string(que.size()) + " Using "; 
    msg += std::to_string(threads_) + " Workers"; 
    this -> info(msg);     

    std::thread* thr_ = nullptr;
    std::vector<size_t>       th_prg(que.size(), 0); 
    std::vector<size_t>       num_data(que.size(), smpls);
    std::vector<std::thread*> th_prc(que.size(), nullptr); 
    std::vector<std::string*> mdl_title(que.size(), nullptr); 

    size_t para = 0; 
    if (!thr_){thr_ = new std::thread(this -> progressbar3, &th_prg, &num_data, &mdl_title);}
    thr_ -> detach(); 
    for (size_t x(0); x < que.size(); ++x){
        que.at(x) -> metrx -> execute(que.at(x), &th_prg[x], mdl_title[x]); 
        std::cout << "--------" << std::endl; 
//        while (para > threads_){para = this -> running(&th_prc, &th_prg, &num_data);} 
    } 
    if (!thr_){this -> failure("No model metrics were executed..."); return false;}
    this -> success("Model metrics completed!"); 
    this -> mflush(&batch_cache); 
    return true; 
}


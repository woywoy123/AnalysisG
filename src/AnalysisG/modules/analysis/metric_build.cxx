#include <structs/switchboards.h>
#include <AnalysisG/analysis.h>
#include <tools/merge_cast.h>
#include <tools/tools.h>

void analysis::build_metric_folds(){
    if (!this -> model_metrics.size()){return;}
    std::vector<int> kfolds = {};
    std::map<std::string, metric_template*>::iterator itm = this -> metric_names.begin();
    for (; itm != this -> metric_names.end(); ++itm){
        std::vector<int> kf = itm -> second -> get_kfolds(); 
        this -> unique_key(&kf, &kfolds); 
    }
    for (size_t x(0); x < kfolds.size(); ++x){kfolds[x];}
    this -> m_settings.kfold = kfolds;
}

bool analysis::build_metric(){
    auto lamb =[this](
            bool mode, mode_enum mt, metric_model_t* mx, 
            std::map<std::string, std::vector<graph_t*>*>* cx
    ) -> long {
        if (!mode){return 0;}
        std::string key = "<"; 
        key += "|" + mx -> run_name; 
        key += "|device-" + std::to_string(mx -> device); 
        key += "|model-mode:" + model_mode(mt); 
        key += "|kfold-" + std::to_string(mx -> kfold); 
        key += "|>"; 
        if (cx -> count(key)){mx -> batches[mt] = (*cx)[key]; return 0;}
        std::vector<graph_t*>* smpl = nullptr; 
        switch(mt){
            case mode_enum::training:   smpl = this -> loader -> get_k_train_set(mx -> kfold - 1);      break;
            case mode_enum::validation: smpl = this -> loader -> get_k_validation_set(mx -> kfold - 1); break;
            case mode_enum::evaluation: smpl = this -> loader -> get_test_set();                        break;
            default: break;   
        }
        if (!smpl){return 0;}
        (*cx)[key] = this -> loader -> build_batch(smpl, mx -> model, nullptr); 
        mx -> batches[mt] = (*cx)[key];
        return long((*cx)[key] -> size()); 
    }; 

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
    abort(); 
    
    long smpls = 0; 
    std::vector<metric_model_t*> que = {}; 
    std::map<int, torch::TensorOptions*> dev_map; 
    std::map<std::string, metric_template*>::iterator itm = this -> metric_names.begin();
    for (; itm != this -> metric_names.end(); ++itm){
        metric_template* mt = itm -> second; 

        // ------------- Get the devices -------------- //
        std::map<int, torch::TensorOptions*> dev_ = mt -> get_devices(); 
        std::map<int, torch::TensorOptions*>::iterator itt; 
        for (itt = dev_.begin(); itt != dev_.end(); ++itt){
            if (dev_map.count(itt -> first)){continue;}
            dev_map[itt -> first] = itt -> second;
        }
        for (size_t x(0); x < mt -> data -> size(); ++x){
            metric_model_t* wrk = mt -> data -> at(x); 
            if ( !wrk -> verify() ){wrk -> failure("ERROR"); continue;}
            wrk -> metrx = mt; que.push_back(wrk); 
        } 
    } 
    this -> loader -> datatransfer(&dev_map);
    multithreaded_t* thr = this -> make_threads(que.size(), threads_); 
        
    // ------------------ Begin the loop ------------------- //
    std::map<std::string, std::vector<graph_t*>*> batch_cache = {}; 
    for (size_t x(0); x < que.size(); ++x){
        metric_model_t* wrk = que[x];  
        size_t tf = 0; 
        tf += lamb(tr, mode_enum::training  , wrk, &batch_cache); 
        tf += lamb(va, mode_enum::validation, wrk, &batch_cache); 
        tf += lamb(ev, mode_enum::evaluation, wrk, &batch_cache); 
        //tracing_t* th = thr -> traces -> at(x); 
        //th -> register_thread(new std::thread(wrk -> metrx -> execute, wrk, th), tf); 
        //while (await_threads(thr, true)){this -> rate_time(1);}
        smpls += tf; 
    }

    std::string msg = "Total Number Events: "; 
    msg += std::to_string(smpls) + " of Jobs Assigned: "; 
    msg += std::to_string(que.size()) + " Using "; 
    msg += std::to_string(threads_) + " Workers"; 
    this -> info(msg);     
}

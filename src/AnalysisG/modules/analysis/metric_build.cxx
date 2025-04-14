#include <AnalysisG/analysis.h>


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
            bool mode, mode_enum mt, std::string hx, int k,
            metric_template* mx, model_template* mdl,
            std::map<std::string, std::vector<graph_t*>*>* cx
    ){
        if (!mode){return;}
        if (cx -> count(hx)){mx -> link(hx, (*cx)[hx], mt); return;}
        std::vector<graph_t*>* smpl = nullptr; 
        switch(mt){
            case mode_enum::training:   smpl = this -> loader -> get_k_train_set(k);      break;
            case mode_enum::validation: smpl = this -> loader -> get_k_validation_set(k); break;
            case mode_enum::evaluation: smpl = this -> loader -> get_test_set();          break;
            default: break;   
        }
        if (!smpl){return;}
        smpl = this -> loader -> build_batch(smpl, mdl, nullptr); 
        (*cx)[hx] = smpl; 
        mx -> link(hx, smpl, mt);
    }; 

    auto lambd = [this](std::map<std::string, std::vector<graph_t*>*>* gr){
        std::map<std::string, std::vector<graph_t*>*>::iterator itg = gr -> begin(); 
        for (; itg != gr -> end(); ++itg){this -> loader -> safe_delete(itg -> second);}
        gr -> clear(); 
    };

    if (!this -> model_metrics.size()){return true;}
    if (!this -> loader -> data_set -> size()){
        this -> failure("No Dataset was found for metrics. Aborting...");
        return false; 
    }

    std::map<int, torch::TensorOptions*> dev_map; 
    std::map<std::string, metric_template*>::iterator itm = this -> metric_names.begin();
    for (; itm != this -> metric_names.end(); ++itm){
        // ------------- Get the devices -------------- //
        std::map<int, torch::TensorOptions*>::iterator itt; 
        std::map<int, torch::TensorOptions*> dev_ = itm -> second -> get_devices(); 
        for (itt = dev_.begin(); itt != dev_.end(); ++itt){
            if (dev_map.count(itt -> first)){continue;}
            dev_map[itt -> first] = itt -> second;
        }
    }
    this -> build_dataloader(true);
    this -> loader -> datatransfer(&dev_map); 

    // ------------- Compute the device and kfold hash -------------- //
    std::map<std::string, std::vector<graph_t*>*> tr_batch_cache = {}; 
    std::map<std::string, std::vector<graph_t*>*> va_batch_cache = {}; 
    std::map<std::string, std::vector<graph_t*>*> ts_batch_cache = {}; 
    for (itm = this -> metric_names.begin(); itm != this -> metric_names.end(); ++itm){
        std::map<std::string, std::vector<model_template*>> mdlx = itm -> second -> hash_mdl; 
        std::map<int, torch::TensorOptions*> dev_ = itm -> second -> get_devices(); 
        std::vector<int> kf = itm -> second -> get_kfolds(); 

        std::map<int, torch::TensorOptions*>::iterator itt; 
        for (itt = dev_.begin(); itt != dev_.end(); ++itt){
            std::string dev_khx = std::to_string(itt -> first) + "+"; 
            for (size_t k(0); k < kf.size(); ++k){
                std::string fx = dev_khx + std::to_string(kf[k]); 
                fx = this -> hash(fx); 
                if (!mdlx.count(fx)){continue;}
                lamb(this -> m_settings.training  , mode_enum::training  , fx, kf[k], itm -> second, mdlx[fx][0], &tr_batch_cache); 
                lamb(this -> m_settings.validation, mode_enum::validation, fx, kf[k], itm -> second, mdlx[fx][0], &va_batch_cache); 
                lamb(this -> m_settings.evaluation, mode_enum::evaluation, fx, kf[k], itm -> second, mdlx[fx][0], &ts_batch_cache); 
            }
        }
    }

    this -> build_project();
    size_t sx = 0; 
    for (itm = this -> metric_names.begin(); itm != this -> metric_names.end(); ++itm){sx += itm -> second -> size();}
    std::vector<size_t> th_prg(sx, 0); 
    std::vector<size_t> num_data(sx, 0);
    std::vector<metric_t*> mx(sx, nullptr); 
    std::vector<std::thread*> th_prc(sx, nullptr); 
    std::vector<metric_template*> mtx(sx, nullptr);
    std::vector<std::string*> th_title(sx, nullptr); 

    sx = 0; 
    for (itm = this -> metric_names.begin(); itm != this -> metric_names.end(); ++itm){
        size_t xt = sx; 
        itm -> second -> define(&mx, &num_data, &th_title, &sx);
        for (; xt < sx; ++xt){mtx[xt] = itm -> second;}
    }

    sx = 0; 
    std::thread* thr_ = nullptr; 
    int threads_ = this -> m_settings.threads; 
    bool debug_mode = this -> m_settings.debug_mode;  
    this -> loader -> start_cuda_server(); 
    for (size_t x(0); x < mx.size(); ++x, ++sx){
        if (debug_mode){this -> execution_metric(mx[x], mtx[x], &th_prg[x], th_title[x]); continue;}
        th_prc[x] = new std::thread(this -> execution_metric, mx[x], mtx[x], &th_prg[x], th_title[x]);
        if (!thr_){thr_ = new std::thread(this -> progressbar3, &th_prg, &num_data, &th_title);}
        while (sx >= threads_){sx = this -> running(&th_prc);} 
    }
    monitor(&th_prc); 
    lambd(&tr_batch_cache); 
    lambd(&va_batch_cache);
    lambd(&ts_batch_cache); 
    if (debug_mode){for (size_t x(0); x < th_title.size(); ++x){delete th_title[x];}}
    if (!thr_){this -> failure("No model metrics were executed..."); return false;}
    thr_ -> join(); delete thr_; thr_ = nullptr; 
    this -> success("Model metrics completed!"); 
    return true; 
}



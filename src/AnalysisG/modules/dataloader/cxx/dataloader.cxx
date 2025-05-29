#include <templates/model_template.h>
#include <structs/report.h>
#include <dataloader.h>
#include <chrono>

dataloader::dataloader(){
    this -> prefix = "dataloader";
    this -> data_set   = new std::vector<graph_t*>();
    this -> data_index = new std::vector<int>(); 
    this -> test_set   = new std::vector<int>(); 
    this -> train_set  = new std::vector<int>(); 
}

dataloader::~dataloader(){
    auto flush = [](std::vector<graph_t*>* grx){
        for (size_t x(0); x < grx -> size(); ++x){
            (*grx)[x] -> _purge_all(); 
            delete (*grx)[x]; 
            (*grx)[x] = nullptr;
        }
        grx -> clear();
        grx -> shrink_to_fit(); 
    }; 

    auto flushM = [](std::vector<std::map<std::string, int>*>* data){
        for (size_t x(0); x < data -> size(); ++x){
            delete (*data)[x]; (*data)[x] = nullptr;
        } 
        data -> clear();
    }; 

    flushM(&this -> truth_map_graph);
    flushM(&this -> truth_map_node);
    flushM(&this -> truth_map_edge);

    flushM(&this -> data_map_graph); 
    flushM(&this -> data_map_node);  
    flushM(&this -> data_map_edge);  

    std::map<int, std::vector<int>*>::iterator itr = this -> k_fold_training.begin(); 
    for (; itr != this -> k_fold_training.end(); ++itr){
        delete this -> gr_k_fold_training[itr -> first]; 
        delete itr -> second;
    }

    std::map<int, std::vector<int>*>::iterator itx = this -> k_fold_validation.begin(); 
    for (; itx != this -> k_fold_validation.end(); ++itx){
        delete this -> gr_k_fold_validation[itr -> first]; 
        delete itx -> second;
    }

    delete this -> test_set; 
    delete this -> train_set; 

    std::vector<graph_t*>* data_ = this -> data_set; 
    this -> data_set = nullptr;
    if (this -> cuda_mem){this -> cuda_mem -> join(); delete this -> cuda_mem;}

    delete this -> data_index; 
    flush(data_); 
    delete data_; data_ = nullptr; 

    std::map<int, std::vector<graph_t*>*>::iterator itc = this -> batched_cache.begin(); 
    for (; itc != this -> batched_cache.end(); ++itc){
        flush(itc -> second); delete itc -> second; itc -> second = nullptr; 
    }
    if (this -> gr_test){delete this -> gr_test;}
}

void dataloader::shuffle(std::vector<int>* idx){
    for (size_t x(0); x < 10; ++x){std::shuffle(idx -> begin(), idx -> end(), this -> rnd);}
}

void dataloader::shuffle(std::vector<graph_t*>* idx){
    std::shuffle(idx -> begin(), idx -> end(), this -> rnd); 
}

void dataloader::clean_data_elements(
        std::map<std::string, int>** data_map, 
        std::vector<std::map<std::string, int>*>* loader_map
){
    int hit = -1; 
    std::map<std::string, int>* dd = *data_map; 
    for (size_t x(0); x < loader_map -> size(); ++x){
        std::map<std::string, int>* ld = (*loader_map)[x];
        if (ld -> size() != dd -> size()){continue;}

        size_t same = 0;  
        std::map<std::string, int>::iterator itr; 
        for (itr = ld -> begin(); itr != ld -> end(); ++itr){
            if (!dd -> count(itr -> first)){break;}
            if ((*dd)[itr -> first] != itr -> second){break;}
            ++same;
        }
        if (same != dd -> size()){continue;}
        hit = int(x); break;
    } 
    if (hit < 0){loader_map -> push_back(dd); return; }
    delete *data_map; *data_map = (*loader_map)[hit];
}

void dataloader::extract_data(graph_t* gr){
    this -> clean_data_elements(&gr -> truth_map_graph, &this -> truth_map_graph); 
    this -> clean_data_elements(&gr -> truth_map_node , &this -> truth_map_node);
    this -> clean_data_elements(&gr -> truth_map_edge , &this -> truth_map_edge);
    this -> clean_data_elements(&gr -> data_map_graph , &this -> data_map_graph);
    this -> clean_data_elements(&gr -> data_map_node  , &this -> data_map_node);
    this -> clean_data_elements(&gr -> data_map_edge  , &this -> data_map_edge);
    this -> hash_map[*gr -> hash] = this -> data_set -> size(); 
    this -> data_index -> push_back(this -> data_set -> size()); 
    this -> data_set -> push_back(gr); 
}


void dataloader::datatransfer(torch::TensorOptions* op, size_t* num_ev, size_t* cur_evnt){
    auto lamb = [](std::vector<graph_t*>* data, torch::TensorOptions* _op, size_t* handle){
        for (size_t f(0); f < data -> size(); ++f){
            (*data)[f] -> transfer_to_device(_op); 
            if (!handle){continue;}
            *handle = f+1;
        }
    };

    if (num_ev){*num_ev = this -> data_set -> size();}
    lamb(this -> data_set, op, cur_evnt); 
}

void dataloader::datatransfer(std::map<int, torch::TensorOptions*>* ops){
    auto lamb = [this](torch::TensorOptions* op, size_t* num_ev, size_t* prg){this -> datatransfer(op, num_ev, prg);};

    size_t num_thr = 0;  
    std::vector<std::thread*> trans(ops -> size(), nullptr); 
    std::vector<std::string*> titles(ops -> size(), nullptr); 
    std::vector<size_t> num_events(ops -> size(), 0); 
    std::vector<size_t> prg_events(ops -> size(), 0);

    this -> info("Transferring graphs to device" + std::string((ops -> size() > 1) ? "s" : "")); 
    std::map<int, torch::TensorOptions*>::iterator ito = ops -> begin();
    for (; ito != ops -> end(); ++ito, ++num_thr){
        trans[num_thr]  = new std::thread(lamb, ito -> second, &num_events[num_thr], &prg_events[num_thr]); 
        titles[num_thr] = new std::string("Progress on device:" + std::to_string(ito -> first)); 
    }
    std::thread* thr = new std::thread(this -> progressbar3, &prg_events, &num_events, &titles); 
    this -> monitor(&trans); 
    this -> success("Transfer Complete!"); 
    thr -> join(); delete thr; thr = nullptr; 
}



std::vector<graph_t*>* dataloader::build_batch(std::vector<graph_t*>* _data, model_template* _mdl, model_report* rep){
    auto g_data  = [this](graph_t* d) -> std::map<int, std::vector<torch::Tensor>>* {return &d -> dev_data_graph;};
    auto n_data  = [this](graph_t* d) -> std::map<int, std::vector<torch::Tensor>>* {return &d -> dev_data_node;};
    auto e_data  = [this](graph_t* d) -> std::map<int, std::vector<torch::Tensor>>* {return &d -> dev_data_edge;};
    auto g_truth = [this](graph_t* d) -> std::map<int, std::vector<torch::Tensor>>* {return &d -> dev_truth_graph;}; 
    auto n_truth = [this](graph_t* d) -> std::map<int, std::vector<torch::Tensor>>* {return &d -> dev_truth_node;};
    auto e_truth = [this](graph_t* d) -> std::map<int, std::vector<torch::Tensor>>* {return &d -> dev_truth_edge;};

    auto collect = [this](
            model_template* __mdl,
            std::vector<graph_t*>* __data, 
            std::map<std::string, int>* loc, 
            std::map<int, std::vector<torch::Tensor>>* cnt, 
            std::function<std::map<int, std::vector<torch::Tensor>>* (graph_t*)> fx)
    {
        std::map<int, torch::Tensor> tmp; 
        std::map<std::string, int>::iterator ilx; 
        for (ilx = loc -> begin(); ilx != loc -> end(); ++ilx){      
            std::string key = ilx -> first; 
            std::vector<torch::Tensor> arr;
            arr.reserve(__data -> size()); 
            for (size_t x(0); x < __data -> size(); ++x){
                graph_t* grx = (*__data)[x]; 
                torch::Tensor* val = grx -> return_any(loc, fx(grx), key, __mdl -> device_index);
                if (val){arr.push_back(*val); continue;}
                this -> warning("broken graph!"); 
                this -> warning("Hash: " + std::string(*grx -> hash) + " -> " + *grx -> filename); 
                abort();
            } 
            if (!arr.size()){continue;}
            tmp[ilx -> second] = torch::Tensor(torch::cat(arr, {0}));
        }
        
        int dev_ = (int)__mdl -> m_option -> device().index();  
        for (ilx = loc -> begin(); ilx != loc -> end(); ++ilx){(*cnt)[dev_].push_back(tmp[ilx -> second]);}
    };  

    auto build_graph = [this, g_data, n_data, e_data, g_truth, n_truth, e_truth, collect](
            std::vector<graph_t*>* inpt, 
            std::vector<graph_t*>* out, 
            model_template* __mdl, 
            size_t index, size_t* prg = nullptr
    ){
        torch::TensorOptions* op = __mdl -> m_option; 
        for (size_t x(0); x < inpt -> size(); ++x){
            if ((*inpt)[x] -> preselection){continue;}
            (*inpt)[x] -> in_use = 1;
            (*inpt)[x] -> transfer_to_device(op); 
        }
        graph_t* tr = (*inpt)[0];  

        bool cached = true; 
        graph_t* gr = (*out)[index]; 
        if (!gr){
            cached = false; 
            gr = new graph_t();
            gr -> data_map_graph  = tr -> data_map_graph; 
            gr -> data_map_node   = tr -> data_map_node; 
            gr -> data_map_edge   = tr -> data_map_edge;

            gr -> truth_map_graph = tr -> truth_map_graph; 
            gr -> truth_map_node  = tr -> truth_map_node ;
            gr -> truth_map_edge  = tr -> truth_map_edge ;
        }

        int dev_ = (int)op -> device().index();  

        // observables 
        collect(__mdl, inpt, gr -> data_map_graph, &gr -> dev_data_graph, g_data); 
        collect(__mdl, inpt, gr -> data_map_node,  &gr -> dev_data_node,  n_data); 
        collect(__mdl, inpt, gr -> data_map_edge,  &gr -> dev_data_edge,  e_data); 

        // truth data
        collect(__mdl, inpt, gr -> truth_map_graph, &gr -> dev_truth_graph, g_truth); 
        collect(__mdl, inpt, gr -> truth_map_node,  &gr -> dev_truth_node,  n_truth); 
        collect(__mdl, inpt, gr -> truth_map_edge,  &gr -> dev_truth_edge,  e_truth); 

        int offset_nodes = 0; 
        std::vector<long> batch_index; 
        std::vector<torch::Tensor> _edge_index;
        std::vector<torch::Tensor> _event_weight; 
        for (size_t x(0); x < inpt -> size(); ++x){
            graph_t* grx = (*inpt)[x]; 
            _edge_index.push_back((*grx -> get_edge_index(__mdl)) + offset_nodes);
            for (int t(0); t < grx -> num_nodes; ++t){batch_index.push_back(x);}
            _event_weight.push_back(*grx -> get_event_weight(__mdl)); 
            offset_nodes += grx -> num_nodes; 
            gr -> in_use = 0; 
            if (cached){continue;}
            gr -> batched_events.push_back(x);
            gr -> batched_filenames.push_back(grx -> filename);
        }

        if (!cached){
            gr -> num_nodes = offset_nodes; 
            gr -> device = op -> device().type(); 
        }

        torch::TensorOptions opx = torch::TensorOptions(torch::kCPU).dtype(torch::kLong); 
        torch::Tensor bx = torch::from_blob(batch_index.data(), {offset_nodes}, opx); 
        torch::Tensor bi = torch::from_blob(gr -> batched_events.data(), {(long)inpt -> size()}, opx); 

        gr -> dev_edge_index[dev_]     = torch::cat(_edge_index, {-1}); 
        gr -> dev_event_weight[dev_]   = torch::cat(_event_weight, {0}); 
        gr -> dev_batch_index[dev_]    = bx.clone().to(op -> device(), true);
        gr -> dev_batched_events[dev_] = bi.clone().to(op -> device(), true); 
        #ifdef PYC_CUDA
        torch::cuda::synchronize(dev_); 
        #endif
        gr -> device_index[dev_] = true; 
        (*out)[index] = gr; 
        if (!prg){return;}
        *prg = 1;
    }; 

    int k   = _mdl -> kfold-1; 
    int dev = _mdl -> m_option -> device().index(); 

    std::vector<graph_t*>* out = nullptr; 
    if (rep && rep -> mode == "evaluation"){k = -1;}

    std::vector<std::vector<graph_t*>> batched = this -> discretize(_data, this -> setting -> batch_size); 
    int thr = this -> setting -> threads; 
    bool skip = thr > int(batched.size()); 

    if (rep && (rep -> mode == "validation" || rep -> mode == "evaluation") && this -> batched_cache.count(k)){
        out = this -> batched_cache[k];
        if (!out -> size()){return out;}
        if ((*out)[0] -> device_index[dev]){return out;}
    }
    else {out = new std::vector<graph_t*>(batched.size(), nullptr);}

    int r = 0; 
    std::vector<size_t> trgt(batched.size(), 1);
    std::vector<size_t> prg(batched.size(), 0);

    std::vector<std::thread*> th_(batched.size(), nullptr); 
    for (size_t x(0); x < batched.size(); ++x, ++r){
        if (skip){build_graph(&batched[x], out, _mdl, x); continue;}
        th_[x] = new std::thread(build_graph, &batched[x], out, _mdl, x, &prg[x]);
        while (r >= thr){r = this -> running(&th_, &prg, &trgt);}
    }
    this -> monitor(&th_); 

    if (!rep){}
    else if (rep -> mode == "validation"){this -> batched_cache[k] = out;}
    else if (rep -> mode == "evaluation"){this -> batched_cache[-1] = out;}
    return out; 
}

void dataloader::safe_delete(std::vector<graph_t*>* data){
    for (size_t x(0); x < data -> size(); ++x){
        (*data)[x] -> _purge_all(); 
        delete (*data)[x]; 
        (*data)[x] = nullptr;
    }
    data -> clear();
    data -> shrink_to_fit(); 
    delete data; 

    #if _server
    c10::cuda::CUDACachingAllocator::emptyCache();
    #endif
}






void dataloader::cuda_memory_server(){
    auto cuda_memory = [this](int device_i) -> bool {
    #if _server
        CUdevice dev;                                    
        cuDeviceGet(&dev, device_i);                     
        size_t free, total;                              
        cuMemGetInfo(&free, &total);                     
        return 100.0*(total - free)/(double)total > 95;  
    #else 
        return false;                                
    #endif
    };                                                   
    auto check_m = [this](std::map<int, std::vector<torch::Tensor>>* in_memory, bool purge, int device){
        std::map<int, std::vector<torch::Tensor>>::iterator ix; 
        for (ix = in_memory -> begin(); ix != in_memory -> end();){
            if (ix -> first == device){++ix; continue;}
            if (!purge){++ix; continue;}
            ix -> second.clear(); 
            ix = in_memory -> erase(++ix); 
        }
    }; 

    auto check_t = [this](std::map<int, torch::Tensor>* in_memory, bool purge, int device){
        std::map<int, torch::Tensor>::iterator ix; 
        for (ix = in_memory -> begin(); ix != in_memory -> end();){
            if (ix -> first == device){++ix; continue;}
            if (!purge){++ix; continue;}
            ix = in_memory -> erase(++ix); 
        }
    }; 

    auto check_b = [this](std::map<int, bool>* in_memory, bool purge, int device){
        std::map<int, bool>::iterator ix; 
        for (ix = in_memory -> begin(); ix != in_memory -> end();){
            if (ix -> first == device){++ix; continue;}
            if (!purge){++ix; continue;}
            ix = in_memory -> erase(++ix); 
        }
    }; 

    bool trig = false;     
    std::vector<graph_t*>* ptr = this -> data_set; 
    for (size_t x(0); x < ptr -> size(); ++x){
        graph_t* gr = (*ptr)[x];
        std::this_thread::sleep_for(std::chrono::microseconds(1));

        if (gr -> in_use == 1){continue;}
        std::map<int, bool>::iterator itx = gr -> device_index.begin(); 
        for (; itx != gr -> device_index.end(); ++itx){
            int dev = itx -> first; 
            if (!cuda_memory(dev)){continue;}
            trig = true; 
            if (gr -> in_use == 1){break;}
            check_m(&gr -> dev_data_graph  , true, dev); 
            check_m(&gr -> dev_data_node   , true, dev); 
            check_m(&gr -> dev_data_edge   , true, dev); 
            check_m(&gr -> dev_truth_graph , true, dev); 
            check_m(&gr -> dev_truth_node  , true, dev); 
            check_m(&gr -> dev_truth_edge  , true, dev);
            check_t(&gr -> dev_edge_index  , true, dev);  
            check_t(&gr -> dev_batch_index , true, dev);  
            check_t(&gr -> dev_event_weight, true, dev);  
            check_b(&gr -> device_index    , true, dev);  
        }
        gr -> in_use = 1; 
    }

    if (!trig){return;}
    #if _server
    c10::cuda::CUDACachingAllocator::emptyCache();
    #endif


}

void dataloader::start_cuda_server(){
    if (this -> cuda_mem){return;}
    if (!_server){return;}
    auto monitor = [this](){
        this -> info("Starting CUDA server!");
        while (this -> data_set){this -> cuda_memory_server();}
        this -> info("Closing CUDA server!");
    };
    this -> cuda_mem = new std::thread(monitor);
}


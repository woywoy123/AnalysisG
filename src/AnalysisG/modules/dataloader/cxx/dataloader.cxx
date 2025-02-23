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
    auto flush = [this](std::vector<graph_t*>* grx){
        for (size_t x(0); x < grx -> size(); ++x){
            (*grx)[x] -> _purge_all(); 
            delete (*grx)[x]; 
            (*grx)[x] = nullptr;
        }
        grx -> clear();
        grx -> shrink_to_fit(); 
    }; 


    for (size_t x(0); x < this -> truth_map_graph.size(); ++x){delete this -> truth_map_graph[x];}
    for (size_t x(0); x < this -> truth_map_node.size(); ++x){delete this -> truth_map_node[x];}
    for (size_t x(0); x < this -> truth_map_edge.size(); ++x){delete this -> truth_map_edge[x];}

    for (size_t x(0); x < this -> data_map_graph.size(); ++x){delete this -> data_map_graph[x];}
    for (size_t x(0); x < this -> data_map_node.size(); ++x){delete this -> data_map_node[x];}
    for (size_t x(0); x < this -> data_map_edge.size(); ++x){delete this -> data_map_edge[x];}

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


void dataloader::datatransfer(torch::TensorOptions* op, int threads){
    auto lamb = [](std::vector<graph_t*>* data, torch::TensorOptions* op, size_t* handle){
        for (size_t f(0); f < data -> size(); ++f){(*data)[f] -> transfer_to_device(op); *handle = f+1;}
    };

    int x = this -> data_set -> size();
    if (!x){return;}
    if (x < threads){threads = 1;}
    x = this -> data_set -> size()/threads; 
    std::vector<std::vector<graph_t*>> quant = this -> discretize(this -> data_set, x); 

    std::vector<size_t> handles(quant.size(), 0); 
    std::vector<std::thread*> th(quant.size(), nullptr);
    for (size_t g(0); g < th.size(); ++g){th[g] = new std::thread(lamb, &quant[g], op, &handles[g]);}

    std::string msg = "Transferring data to device."; 
    std::thread* prg = new std::thread(this -> progressbar1, &handles, this -> data_set -> size(), msg);
    for (size_t g(0); g < th.size(); ++g){th[g] -> join(); delete th[g];}
    prg -> join(); delete prg; 
}

std::vector<graph_t*>* dataloader::build_batch(std::vector<graph_t*>* data, model_template* mdl, model_report* rep){
    auto g_data  = [this](graph_t* d) -> std::map<int, std::vector<torch::Tensor>>* {return &d -> dev_data_graph;};
    auto n_data  = [this](graph_t* d) -> std::map<int, std::vector<torch::Tensor>>* {return &d -> dev_data_node;};
    auto e_data  = [this](graph_t* d) -> std::map<int, std::vector<torch::Tensor>>* {return &d -> dev_data_edge;};
    auto g_truth = [this](graph_t* d) -> std::map<int, std::vector<torch::Tensor>>* {return &d -> dev_truth_graph;}; 
    auto n_truth = [this](graph_t* d) -> std::map<int, std::vector<torch::Tensor>>* {return &d -> dev_truth_node;};
    auto e_truth = [this](graph_t* d) -> std::map<int, std::vector<torch::Tensor>>* {return &d -> dev_truth_edge;};

    auto collect = [this](
            model_template* mdl,
            std::vector<graph_t*>* data, 
            std::map<std::string, int>* loc, 
            std::map<int, std::vector<torch::Tensor>>* cnt, 
            std::function<std::map<int, std::vector<torch::Tensor>>* (graph_t*)> fx)
    {
        std::map<int, torch::Tensor> tmp; 
        std::map<std::string, int>::iterator ilx; 
        for (ilx = loc -> begin(); ilx != loc -> end(); ++ilx){      
            std::string key = ilx -> first; 
            std::vector<torch::Tensor> arr;
            for (size_t x(0); x < data -> size(); ++x){
                graph_t* grx = (*data)[x]; 
                torch::Tensor* val = grx -> return_any(loc, fx(grx), key, mdl);
                if (!val){continue;} 
                arr.push_back(*val); 
            } 
            if (!arr.size()){continue;}
            tmp[ilx -> second] = torch::Tensor(torch::cat(arr, {0}));
        }
        
        int dev_ = (int)mdl -> m_option -> device().index();  
        for (ilx = loc -> begin(); ilx != loc -> end(); ++ilx){(*cnt)[dev_].push_back(tmp[ilx -> second]);}
    };  

    auto build_graph = [this, g_data, n_data, e_data, g_truth, n_truth, e_truth, collect](
            std::vector<graph_t*>* inpt, 
            std::vector<graph_t*>* out, 
            model_template* mdl, size_t index)
    {
        torch::TensorOptions* op = mdl -> m_option; 
        for (size_t x(0); x < inpt -> size(); ++x){
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
        collect(mdl, inpt, gr -> data_map_graph, &gr -> dev_data_graph, g_data); 
        collect(mdl, inpt, gr -> data_map_node,  &gr -> dev_data_node,  n_data); 
        collect(mdl, inpt, gr -> data_map_edge,  &gr -> dev_data_edge,  e_data); 

        // truth data
        collect(mdl, inpt, gr -> truth_map_graph, &gr -> dev_truth_graph, g_truth); 
        collect(mdl, inpt, gr -> truth_map_node,  &gr -> dev_truth_node,  n_truth); 
        collect(mdl, inpt, gr -> truth_map_edge,  &gr -> dev_truth_edge,  e_truth); 

        int offset_nodes = 0; 
        std::vector<long> batch_index; 
        std::vector<torch::Tensor> _edge_index;
        std::vector<torch::Tensor> _event_weight; 
        for (size_t x(0); x < inpt -> size(); ++x){
            graph_t* grx = (*inpt)[x]; 
            _edge_index.push_back((*grx -> get_edge_index(mdl)) + offset_nodes);
            for (int t(0); t < grx -> num_nodes; ++t){batch_index.push_back(x);}
            _event_weight.push_back(*grx -> get_event_weight(mdl)); 
            offset_nodes += grx -> num_nodes; 
            gr -> in_use = 0; 
            if (cached){continue;}
            gr -> batched_events.push_back(x);
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
        gr -> dev_batch_index[dev_]    = bx.clone().to(op -> device());
        gr -> dev_batched_events[dev_] = bi.clone().to(op -> device()); 
        gr -> device_index[dev_] = true; 

        (*out)[index] = gr; 
        torch::cuda::synchronize(); 
    }; 

    int k = mdl -> kfold-1; 
    int dev = mdl -> m_option -> device().index(); 

    std::vector<graph_t*>* out = nullptr; 
    if (rep && rep -> mode == "evaluation"){k = -1;}

    std::vector<std::vector<graph_t*>> batched = this -> discretize(data, this -> setting -> batch_size); 
    int thr = this -> setting -> threads; 
    bool skip = bool(thr > batched.size()); 

    if (rep && (rep -> mode == "validation" || rep -> mode == "evaluation") && this -> batched_cache.count(k)){
        out = this -> batched_cache[k];
        if (out -> at(0) -> device_index[dev]){return out;}
    }
    else {out = new std::vector<graph_t*>(batched.size(), nullptr);}

    int r = 0; 
    std::vector<std::thread*> th_(batched.size(), nullptr); 
    for (size_t x(0); x < batched.size(); ++x){
        if (skip){build_graph(&batched[x], out, mdl, x); continue;}
        th_[x] = new std::thread(build_graph, &batched[x], out, mdl, x);
        ++r; 
        while (r >= thr){
            for (size_t i(0); i < x; ++i){
                if (!th_[i]){continue;}
                if (!th_[i] -> joinable()){continue;}
                th_[i] -> join(); 
                delete th_[i]; th_[i] = nullptr; 
                --r; 
                break;
            }
        }
    }

    for (size_t i(0); i < batched.size(); ++i){
        if (!th_[i]){continue;}
        if (!th_[i] -> joinable()){continue;}
        th_[i] -> join(); 
        delete th_[i]; th_[i] = nullptr; 
    }

    if (!rep){}
    else if (rep -> mode == "validation"){this -> batched_cache[k] = out;}
    else if (rep -> mode == "evaluation"){this -> batched_cache[-1] = out;}
    return out; 
}

void dataloader::cuda_memory_server(){
    auto cuda_memory = [this](int device_i) -> bool {
    #if server
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
    
    std::vector<graph_t*>* ptr = this -> data_set; 
    #if server
    c10::cuda::CUDACachingAllocator::emptyCache();
    #endif
    for (size_t x(0); x < ptr -> size(); ++x){
        graph_t* gr = (*ptr)[x];
        std::this_thread::sleep_for(std::chrono::microseconds(1));

        if (gr -> in_use == 1){continue;}
        std::map<int, bool>::iterator itx = gr -> device_index.begin(); 
        for (; itx != gr -> device_index.end(); ++itx){
            int dev = itx -> first; 
            if (!cuda_memory(dev)){continue;}
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
}

void dataloader::start_cuda_server(){
    if (!server){return;}
    if (this -> cuda_mem){return;}
    auto monitor = [this](){
        this -> info("Starting CUDA server!");
        while (this -> data_set){this -> cuda_memory_server();}
        this -> info("Closing CUDA server!");
    };
    this -> cuda_mem = new std::thread(monitor);
}


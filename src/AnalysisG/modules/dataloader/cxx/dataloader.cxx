#include <chrono>
#include <generators/dataloader.h>
#include <structs/folds.h>
#include <io/io.h>
#include <thread>

dataloader::dataloader(){
    this -> prefix = "dataloader";
    this -> data_set   = new std::vector<graph_t*>();
    this -> data_index = new std::vector<int>(); 
    this -> test_set   = new std::vector<int>(); 
    this -> train_set  = new std::vector<int>(); 
    if (!this -> cuda_mem){return;}
    this -> cuda_mem -> join(); 
    delete this -> cuda_mem; 
}

dataloader::~dataloader(){
    for (size_t x(0); x < this -> truth_map_graph.size(); ++x){delete this -> truth_map_graph[x];}
    for (size_t x(0); x < this -> truth_map_node.size(); ++x){delete this -> truth_map_node[x];}
    for (size_t x(0); x < this -> truth_map_edge.size(); ++x){delete this -> truth_map_edge[x];}

    for (size_t x(0); x < this -> data_map_graph.size(); ++x){delete this -> data_map_graph[x];}
    for (size_t x(0); x < this -> data_map_node.size(); ++x){delete this -> data_map_node[x];}
    for (size_t x(0); x < this -> data_map_edge.size(); ++x){delete this -> data_map_edge[x];}

    std::map<int, std::vector<int>*>::iterator itr = this -> k_fold_training.begin(); 
    for (; itr != this -> k_fold_training.end(); ++itr){delete itr -> second;}

    std::map<int, std::vector<int>*>::iterator itx = this -> k_fold_validation.begin(); 
    for (; itx != this -> k_fold_validation.end(); ++itx){delete itx -> second;}

    delete this -> test_set; 
    delete this -> train_set; 

    delete this -> data_index; 
    for (size_t x(0); x < this -> data_set -> size(); ++x){
        this -> data_set -> at(x) -> _purge_all(); 
        delete this -> data_set -> at(x); 
    }
    delete this -> data_set; 
}

void dataloader::generate_kfold_set(int k){
    if (!this -> test_set -> size()){this -> shuffle(this -> data_index);}

    bool all = false;
    for (int x(0); x < k; ++x){
        if (this -> k_fold_training.count(x)){continue;}
        this -> k_fold_training[x] = new std::vector<int>(); 
        this -> k_fold_validation[x] = new std::vector<int>();
        all = true; 
    }
    if (!all){return;}
    std::map<int, std::vector<int>> folds = {}; 
    for (int x(0); x < this -> train_set -> size(); ++x){
        folds[x%(k+1)].push_back((*this -> train_set)[x]);
    }
    this -> success("Splitting training dataset (" + this -> to_string(this -> train_set -> size()) + ")"); 
    for (int x(0); x < k; ++x){
        std::vector<int>* val = this -> k_fold_validation[x]; 
        val -> insert(val -> end(), folds[x].begin(), folds[x].end()); 
        for (int y(0); y < k+1; ++y){
            if (y == x){continue;}
            val = this -> k_fold_training[x]; 
            val -> insert(val -> end(), folds[y].begin(), folds[y].end()); 
        }
        this -> success("---------------- k-Fold: " + this -> to_string(x+1) + " ----------------"); 
        this -> success("-> train: " + this -> to_string(this -> k_fold_training[x] -> size()) + ")"); 
        this -> success("-> validation: " + this -> to_string(this -> k_fold_validation[x] -> size()) + ")"); 
    }
}

void dataloader::dump_dataset(std::string path){
    io* io_g = new io(); 
    std::vector<folds_t> data = {};  
    std::map<int, std::vector<int>*> data_e; 
    std::map<int, std::vector<int>*>::iterator itr; 

    data_e = this -> k_fold_training; 
    for (itr = data_e.begin(); itr != data_e.end(); ++itr){
        size_t len = itr -> second -> size(); 
        for (size_t x(0); x < len; ++x){
            folds_t kf = folds_t(); 
            kf.k = itr -> first;
            kf.is_train = true; 
            kf.index = (*itr -> second)[x]; 
            data.push_back(kf); 
        }
    } 

    data_e = this -> k_fold_validation; 
    for (itr = data_e.begin(); itr != data_e.end(); ++itr){
        size_t len = itr -> second -> size(); 
        for (size_t x(0); x < len; ++x){
            folds_t kf = folds_t(); 
            kf.k = itr -> first;
            kf.is_valid = true; 
            kf.index = (*itr -> second)[x]; 
            data.push_back(kf); 
        }
    } 

    for (size_t x(0); x < this -> test_set -> size(); ++x){
        folds_t kf = folds_t(); 
        kf.is_eval = true; 
        kf.index = (*this -> test_set)[x]; 
        data.push_back(kf); 
    }
    io_g -> start(path, "write"); 
    io_g -> write(&data, "kfolds"); 
    io_g -> end(); 

    delete io_g; 
}

bool dataloader::restore_dataset(std::string path){
    if (this -> k_fold_training.size()){return true;}
    io* io_g = new io(); 
    io_g -> start(path, "read"); 
    std::vector<folds_t> data = {}; 
    io_g -> read(&data, "kfolds"); 
    io_g -> end(); 
    delete io_g; 

    for (size_t x(0); x < data.size(); ++x){
        folds_t* kf = &data[x]; 
        if (kf -> is_eval){this -> test_set -> push_back(kf -> index);continue;}
        if (kf -> k == 0){this -> train_set -> push_back(kf -> index);}
        if (!this -> k_fold_training.count(kf -> k)){
            this -> k_fold_training[kf -> k]   = new std::vector<int>();
            this -> k_fold_validation[kf -> k] = new std::vector<int>(); 
        }

        std::vector<int>* bin = nullptr; 
        if (kf -> is_train){bin = this -> k_fold_training[kf -> k];}
        else {bin = this -> k_fold_validation[kf -> k];}
        bin -> push_back(kf -> index); 
    }
    if (!data.size()){return false;}

    this -> success("Restored training dataset (" + this -> to_string(this -> train_set -> size()) + ")"); 
    std::map<int, std::vector<int>*>::iterator itr = this -> k_fold_training.begin(); 
    for (; itr != this -> k_fold_training.end(); ++itr){
        int k = itr -> first; 
        std::vector<int>* val = itr -> second; 
        this -> success("---------------- k-Fold: " + this -> to_string(k+1) + " ----------------"); 
        this -> success("-> train: " + this -> to_string(this -> k_fold_training[k] -> size()) + ")"); 
        this -> success("-> validation: " + this -> to_string(this -> k_fold_validation[k] -> size()) + ")"); 
    }
    return true;
}

void dataloader::generate_test_set(float percentage){
    if (this -> test_set -> size()){return;}
    this -> data_set    -> shrink_to_fit(); 
    this -> data_index  -> shrink_to_fit();   

    int fx = (this -> data_set -> size()) * float(percentage/100); 
    this -> shuffle(this -> data_index); 
    for (size_t x(0); x < this -> data_index -> size(); ++x){
        std::vector<int>* dx = nullptr; 
        if (x >= fx){dx = this -> test_set;}
        else {dx = this -> train_set;}
        dx -> push_back(this -> data_index -> at(x));
    }
    this -> test_set -> shrink_to_fit();
    this -> train_set -> shrink_to_fit(); 
    this -> success("Splitting entire dataset (" + this -> to_string(this -> data_set -> size()) + ")"); 
    this -> success("-> test: " + this -> to_string(this -> test_set -> size()) + ")"); 
    this -> success("-> train: " + this -> to_string(this -> train_set -> size()) + ")"); 
}

void dataloader::shuffle(std::vector<int>* idx){
    std::shuffle(idx -> begin(), idx -> end(), this -> rnd); 
}

void dataloader::clean_data_elements(
                std::map<std::string, int>** data_map, 
                std::vector<std::map<std::string, int>*>* loader_map
){
    int hit = -1; 
    std::map<std::string, int>* dd = *data_map; 
    for (int x(0); x < loader_map -> size(); ++x){
        std::map<std::string, int>* ld = loader_map -> at(x); 
        if (ld -> size() != dd -> size()){continue;}

        int same = 0;  
        std::map<std::string, int>::iterator itr; 
        for (itr = ld -> begin(); itr != ld -> end(); ++itr){
            if (!dd -> count(itr -> first)){break;}
            if ((*dd)[itr -> first] != itr -> second){break;}
            ++same;
        }
        if (same != dd -> size()){continue;}
        hit = x; break;
    } 
    if (hit < 0){loader_map -> push_back(dd); return; }
    delete *data_map; 
    *data_map = loader_map -> at(hit); 
}

std::vector<graph_t*> dataloader::get_random(int num){
    this -> shuffle(this -> data_index); 
    std::vector<graph_t*> out = {}; 
    for (int k(0); k < num; ++k){
        int n = this -> data_index -> at(k); 
        out.push_back(this -> data_set -> at(n)); 
    }
    return out; 
}

std::vector<graph_t*>* dataloader::get_k_train_set(int k){
    if (this -> gr_k_fold_training.count(k)){return this -> gr_k_fold_training[k];}
    if (!this -> k_fold_training.count(k)){
        this -> warning("Specified an invalid k-fold index."); 
        return nullptr;
    }
    
    std::vector<int>* kdata = this -> k_fold_training[k]; 
    this -> shuffle(kdata); 
    std::vector<graph_t*>* output = new std::vector<graph_t*>();
    for (int x(0); x < kdata -> size(); ++x){
        graph_t* gr = (*this -> data_set)[(*kdata)[x]]; 
        output -> push_back(gr);
        gr -> in_use = 1; 
    }
    output -> shrink_to_fit(); 
    this -> gr_k_fold_training[k] = output; 
    return output; 
}


std::vector<graph_t*>* dataloader::get_k_validation_set(int k){
    if (this -> gr_k_fold_validation.count(k)){return this -> gr_k_fold_validation[k];}

    std::vector<int>* kdata = this -> k_fold_validation[k]; 
    this -> shuffle(kdata); 
    std::vector<graph_t*>* output = new std::vector<graph_t*>();
    for (int x(0); x < kdata -> size(); ++x){
        graph_t* gr = (*this -> data_set)[(*kdata)[x]]; 
        output -> push_back(gr);
        gr -> in_use = 1; 
    }
    output -> shrink_to_fit(); 
    this -> gr_k_fold_validation[k] = output; 
    return output; 
}

std::vector<graph_t*>* dataloader::get_test_set(){
    if (this -> gr_test){return this -> gr_test;}
    this -> shuffle(this -> test_set); 
    std::vector<graph_t*>* output = new std::vector<graph_t*>();
    for (int x(0); x < this -> test_set -> size(); ++x){
        graph_t* gr = (*this -> data_set)[(*this -> test_set)[x]]; 
        output -> push_back(gr);
        gr -> in_use = 1; 
    }
    output -> shrink_to_fit(); 
    this -> gr_test = output; 
    return output; 
}

std::map<std::string, std::vector<graph_t*>>* dataloader::get_inference(){
    auto lamb = [](std::vector<graph_t*>* sort){
        std::map<long, graph_t*> tmp = {}; 
        for (size_t x(0); x < sort -> size(); ++x){tmp[(*sort)[x] -> event_index] = (*sort)[x];}

        int t = 0; 
        std::map<long, graph_t*>::iterator itr = tmp.begin(); 
        for (; itr != tmp.end(); ++itr, ++t){(*sort)[t] = itr -> second;}
    }; 


    std::map<std::string, std::vector<graph_t*>>* out = new std::map<std::string, std::vector<graph_t*>>();
    for (size_t x(0); x < this -> data_set -> size(); ++x){
        graph_t* gr = this -> data_set -> at(x); 
        (*out)[*(gr -> filename)].push_back(gr); 
    }

    int t = 0; 
    std::vector<std::thread*> th(out -> size(), nullptr); 
    std::map<std::string, std::vector<graph_t*>>::iterator itr = out -> begin(); 
    for (; itr != out -> end(); ++itr, ++t){th[t] = new std::thread(lamb, &itr -> second);}
    for (t = 0; t < th.size(); ++t){th[t] -> join(); delete th[t]; th[t] = nullptr;}
    return out; 
}

void dataloader::extract_data(graph_t* gr){
    this -> clean_data_elements(&gr -> truth_map_graph, &this -> truth_map_graph); 
    this -> clean_data_elements(&gr -> truth_map_node , &this -> truth_map_node);
    this -> clean_data_elements(&gr -> truth_map_edge , &this -> truth_map_edge);
    this -> clean_data_elements(&gr -> data_map_graph , &this -> data_map_graph);
    this -> clean_data_elements(&gr -> data_map_node  , &this -> data_map_node);
    this -> clean_data_elements(&gr -> data_map_edge  , &this -> data_map_edge);
    this -> data_set -> push_back(gr); 
    this -> data_index -> push_back(this -> data_index -> size()); 
}


void dataloader::datatransfer(torch::TensorOptions* op, int threads){
    auto lamb = [](std::vector<graph_t*>* data, torch::TensorOptions* op){
        for (size_t f(0); f < data -> size(); ++f){(*data)[f] -> transfer_to_device(op);}
    };

    if (!this -> data_set){return;}
    int x = this -> data_set -> size()/threads; 
    std::vector<std::vector<graph_t*>> quant = this -> discretize(this -> data_set, x); 
    std::vector<std::thread*> th(quant.size(), nullptr);
    for (size_t g(0); g < th.size(); ++g){th[g] = new std::thread(lamb, &quant[g], op);}

    std::string msg = "Transferring data to device."; 
    this -> progressbar(0, msg); 
    for (size_t g(0); g < th.size(); ++g){
        th[g] -> join(); delete th[g];
        this -> progressbar(float(g+1)/float(th.size()), msg); 
    }
    std::cout << "" << std::endl;
    if (this -> tensor_op){return;}
    this -> tensor_op = op; 
}

void dataloader::cuda_memory_server(){
    if (!this -> tensor_op){return;}
    torch::TensorOptions* op = new torch::TensorOptions(c10::kCPU);
    int id = this-> tensor_op -> device().index(); 

    CUdevice dev; 
    cuDeviceGet(&dev, id); 

    size_t free, total; 
    cuMemGetInfo(&free, &total); 

    double perc = 100.0*(total - free)/(double)total; 
    bool full_purge = perc > 95; 
    for (size_t x(0); x < this -> data_set -> size(); ++x){
        std::this_thread::sleep_for(std::chrono::microseconds(10));
        graph_t* gr = (*this -> data_set)[x]; 
        if (gr -> in_use == 2){continue;}
        if (gr -> in_use == 1 && !full_purge){continue;}
        if (gr -> in_use == -1){continue;}
        gr -> in_use = 0; 
        gr -> transfer_to_device(op);  
        gr -> in_use = -1; 
    }
    delete op; 
}

void dataloader::start_cuda_server(){
    if (this -> cuda_mem){return;}
    auto monitor = [this](){
        while (this -> data_set){
            this -> cuda_memory_server(); 
        }
    }; 
    this -> cuda_mem = new std::thread(monitor);
}

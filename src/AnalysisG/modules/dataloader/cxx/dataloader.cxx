#include "dataloader.h"


dataloader::dataloader(){
    this -> prefix = "dataloader";
    this -> data_set   = new std::vector<graph_t*>();
    this -> data_index = new std::vector<int>(); 
    this -> test_set   = new std::vector<int>(); 
    this -> train_set  = new std::vector<int>(); 
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

    std::vector<graph_t*>* data_ = this -> data_set; 
    this -> data_set = nullptr;
    delete this -> data_index; 
    for (size_t x(0); x < data_ -> size(); ++x){
        (*data_)[x] -> _purge_all(); 
        delete (*data_)[x]; 
        (*data_)[x] = nullptr;
    }
    data_ -> clear();
    data_ -> shrink_to_fit();
    delete data_; 
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
    for (int x(0); x < loader_map -> size(); ++x){
        std::map<std::string, int>* ld = (*loader_map)[x];
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
    *data_map = (*loader_map)[hit];
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

#include "dataloader.h"
#include <structs/folds.h>
#include <io/io.h>

std::vector<graph_t*> dataloader::get_random(int num){
    this -> shuffle(this -> data_index); 
    std::vector<graph_t*> out = {}; 
    for (int k(0); k < num; ++k){
        int n = (*this -> data_index)[k]; 
        out.push_back((*this -> data_set)[n]); 
    }
    return out; 
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

std::vector<graph_t*>* dataloader::get_k_train_set(int k){
    if (this -> gr_k_fold_training.count(k)){
        this -> shuffle(this -> gr_k_fold_training[k]); 
        return this -> gr_k_fold_training[k];
    }

    if (!this -> k_fold_training.count(k)){
        this -> warning("Specified an invalid k-fold index."); 
        return nullptr;
    }

    std::vector<int>* kdata = this -> k_fold_training[k]; 
    this -> shuffle(kdata); 
    this -> gr_k_fold_training[k] = new std::vector<graph_t*>();
    this -> put(this -> gr_k_fold_training[k], this -> data_set, kdata); 
    this -> gr_k_fold_training[k] -> shrink_to_fit(); 
    return this -> gr_k_fold_training[k]; 
}

std::vector<graph_t*>* dataloader::get_k_validation_set(int k){
    if (this -> gr_k_fold_validation.count(k)){return this -> gr_k_fold_validation[k];}
    std::vector<int>* kdata = this -> k_fold_validation[k]; 
    this -> shuffle(kdata); 
    this -> gr_k_fold_validation[k] = new std::vector<graph_t*>();
    this -> put(this -> gr_k_fold_validation[k], this -> data_set, kdata); 
    this -> gr_k_fold_validation[k] -> shrink_to_fit(); 
    return this -> gr_k_fold_validation[k]; 
}

std::vector<graph_t*>* dataloader::get_test_set(){
    if (this -> gr_test){return this -> gr_test;}
    this -> shuffle(this -> test_set); 

    std::vector<graph_t*>* output = new std::vector<graph_t*>();
    this -> put(output, this -> data_set, test_set); 
    output -> shrink_to_fit(); 

    this -> gr_test = output; 
    return output; 
}

std::map<std::string, std::vector<graph_t*>>* dataloader::get_inference(){
    auto lamb = [](std::vector<graph_t*>* sort){
        std::map<long, graph_t*> tmp = {}; 
        for (size_t x(0); x < sort -> size(); ++x){tmp[(*sort)[x] -> event_index] = (*sort)[x];}

        std::map<long, graph_t*>::iterator itr = tmp.begin(); 
        for (int t(0); itr != tmp.end(); ++itr, ++t){(*sort)[t] = itr -> second;}
    }; 


    std::map<std::string, std::vector<graph_t*>>* out = new std::map<std::string, std::vector<graph_t*>>();
    for (size_t x(0); x < this -> data_set -> size(); ++x){
        graph_t* gr = (*this -> data_set)[x]; 
        (*out)[*(gr -> filename)].push_back(gr); 
    }

    std::vector<std::thread*> th(out -> size(), nullptr); 
    std::map<std::string, std::vector<graph_t*>>::iterator itr = out -> begin(); 
    for (int t(0); itr != out -> end(); ++itr, ++t){th[t] = new std::thread(lamb, &itr -> second);}
    for (int t(0); t < th.size(); ++t){th[t] -> join(); delete th[t]; th[t] = nullptr;}
    return out; 
}


// ................... dataset dumping ........................ //
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
    this -> success("Leave out sample is (" + this -> to_string(this -> test_set -> size()) + ")"); 
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



#include <generators/dataloader.h>

dataloader::dataloader(){
    this -> prefix = "dataloader";
    this -> data_set = new std::vector<graph_t*>();
    this -> data_hashes = new std::vector<std::string>(); 
    this -> data_index = new std::vector<int>(); 
    this -> test_set   = new std::vector<int>(); 
    this -> train_set  = new std::vector<int>(); 
}

dataloader::~dataloader(){}

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
    for (int x(0); x < this -> data_index -> size(); ++x){
        folds[x%k].push_back(this -> data_index -> at(x)); 
    }

    for (int x(0); x < k; ++x){
        std::vector<int>* val = this -> k_fold_validation[x]; 
        val -> insert(val -> end(), folds[x].begin(), folds[x].end()); 
        for (int y(0); y < k; ++y){
            if (y == x){continue;}
            val = this -> k_fold_training[y]; 
            val -> insert(val -> end(), folds[y].begin(), folds[y].end()); 
        }
    }
}

void dataloader::generate_test_set(float percentage){
    this -> data_set    -> shrink_to_fit(); 
    this -> data_hashes -> shrink_to_fit();
    this -> data_index  -> shrink_to_fit();   

    int fx = (this -> data_set -> size()) * int(percentage/100); 
    this -> shuffle(this -> data_index); 

    for (size_t x(0); x < this -> data_index -> size(); ++x){
        std::vector<int>* dx = nullptr; 
        if (x < fx){dx = this -> test_set;}
        else {dx = this -> train_set;}
        dx -> push_back(this -> data_index -> at(x));
    }
    this -> test_set -> shrink_to_fit();
    this -> train_set -> shrink_to_fit(); 
}

void dataloader::shuffle(std::vector<int>* idx){
    std::shuffle(idx -> begin(), idx -> end(), this -> rnd); 
}

void dataloader::clean_data_elements(
                std::map<std::string, int>** data_map, 
                std::map<std::string, int>** loader_map
){
    if (!(*loader_map)){*loader_map = *data_map;}
    else {delete *data_map; *data_map = *loader_map;}
}

void dataloader::extract_data(graph_template* data){
    graph_t* gr = data -> data_export(); 
    this -> clean_data_elements(&gr -> truth_map_graph, &this -> truth_map_graph); 
    this -> clean_data_elements(&gr -> truth_map_node , &this -> truth_map_node);
    this -> clean_data_elements(&gr -> truth_map_edge , &this -> truth_map_edge);
    this -> clean_data_elements(&gr -> data_map_graph , &this -> data_map_graph);
    this -> clean_data_elements(&gr -> data_map_node  , &this -> data_map_node);
    this -> clean_data_elements(&gr -> data_map_edge  , &this -> data_map_edge);
    this -> data_set -> push_back(gr); 
    this -> data_hashes -> push_back(data -> hash); 
    this -> data_index -> push_back(this -> data_index -> size()); 
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


void dataloader::add_to_collection(std::vector<graph_template*>* inpt){
    for (size_t x(0); x < inpt -> size(); ++x){this -> extract_data(inpt -> at(x));}
    for (size_t x(0); x < inpt -> size(); ++x){delete inpt -> at(x);}
    delete inpt; 
}







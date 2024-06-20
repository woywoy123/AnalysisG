#include <generators/dataloader.h>

dataloader::dataloader(){
    this -> prefix = "dataloader";
    this -> data_set = new std::vector<graph_t*>();
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
    for (int x(0); x < k+1; ++x){
        if (this -> k_fold_training.count(x)){continue;}
        this -> k_fold_training[x] = new std::vector<int>(); 
        this -> k_fold_validation[x] = new std::vector<int>();
        all = true; 
    }
    if (!all){return;}
    std::map<int, std::vector<int>> folds = {}; 
    for (int x(0); x < this -> train_set -> size(); ++x){
        folds[x%(k+1)].push_back(this -> train_set -> at(x));
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
    this -> data_set    -> shrink_to_fit(); 
    this -> data_index  -> shrink_to_fit();   

    int fx = (this -> data_set -> size()) * float(percentage/100); 
    this -> shuffle(this -> data_index); 
    for (size_t x(0); x < this -> data_index -> size(); ++x){
        std::vector<int>* dx = nullptr; 
        if (x < fx){dx = this -> test_set;}
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

    std::vector<int>* kdata = this -> k_fold_training[k]; 
    this -> shuffle(kdata); 
    std::vector<graph_t*>* output = new std::vector<graph_t*>();
    for (int x(0); x < kdata -> size(); ++x){
        output -> push_back(this -> data_set -> at(kdata -> at(x)));
    }
    output -> shrink_to_fit(); 
    this -> gr_k_fold_training[k-1] = output; 
    return output; 
}


std::vector<graph_t*>* dataloader::get_k_validation_set(int k){
    if (this -> gr_k_fold_validation.count(k-1)){return this -> gr_k_fold_validation[k-1];}

    std::vector<int>* kdata = this -> k_fold_validation[k]; 
    this -> shuffle(kdata); 
    std::vector<graph_t*>* output = new std::vector<graph_t*>();
    for (int x(0); x < kdata -> size(); ++x){
        output -> push_back(this -> data_set -> at(kdata -> at(x)));
    }
    output -> shrink_to_fit(); 
    this -> gr_k_fold_validation[k-1] = output; 
    return output; 
}

std::vector<graph_t*>* dataloader::get_test_set(){
    if (this -> gr_test){return this -> gr_test;}
    this -> shuffle(this -> test_set); 
    std::vector<graph_t*>* output = new std::vector<graph_t*>();
    for (int x(0); x < this -> test_set -> size(); ++x){
        output -> push_back(this -> data_set -> at(this -> test_set -> at(x)));
    }
    output -> shrink_to_fit(); 
    this -> gr_test = output; 
    return output; 
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

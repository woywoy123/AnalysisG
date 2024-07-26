#include <templates/graph_template.h>


void graph_t::add_truth_graph(std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps){
    if (this -> truth_graph){return;}
    this -> truth_graph = this -> add_content(data); 
    this -> truth_map_graph = maps; 
}

void graph_t::add_truth_node(std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps){
    if (this -> truth_node){return;}
    this -> truth_node = this -> add_content(data);
    this -> truth_map_node = maps; 
}

void graph_t::add_truth_edge(std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps){
    if (this -> truth_edge){return;}
    this -> truth_edge = this -> add_content(data);
    this -> truth_map_edge = maps; 
}

void graph_t::add_data_graph(std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps){
    if (this -> data_graph){return;}
    this -> data_graph = this -> add_content(data); 
    this -> data_map_graph = maps; 
}

void graph_t::add_data_node(std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps){
    if (this -> data_node){return;}
    this -> data_node = this -> add_content(data);
    this -> data_map_node = maps; 
}

void graph_t::add_data_edge(std::map<std::string, torch::Tensor*>* data, std::map<std::string, int>* maps){
    if (this -> data_edge){return;}
    this -> data_edge = this -> add_content(data);
    this -> data_map_edge = maps; 
}

void graph_t::_purge_all(){
    this -> _purge_data(this -> data_graph); 
    this -> _purge_data(this -> data_node); 
    this -> _purge_data(this -> data_edge); 
    delete this -> edge_index;

    this -> _purge_data(this -> truth_graph); 
    this -> _purge_data(this -> truth_node); 
    this -> _purge_data(this -> truth_edge); 

    this -> _purge_data(&this -> dev_data_graph); 
    this -> _purge_data(&this -> dev_data_node); 
    this -> _purge_data(&this -> dev_data_edge); 

    this -> _purge_data(&this -> dev_truth_graph); 
    this -> _purge_data(&this -> dev_truth_node); 
    this -> _purge_data(&this -> dev_truth_edge); 

    std::map<int, torch::Tensor*>::iterator itr = this -> dev_edge_index.begin();
    for (; itr != this -> dev_edge_index.end(); ++itr){delete itr -> second;}
    this -> dev_edge_index.clear(); 
}

void graph_t::_purge_data(std::vector<torch::Tensor*>* data){
    if (!data){return;}
    for (size_t x(0); x < data -> size(); ++x){delete data -> at(x);}
}

void graph_t::_purge_data(std::map<int, std::vector<torch::Tensor*>*>* data){
    std::map<int, std::vector<torch::Tensor*>*>::iterator itr = data -> begin();
    for (; itr != data -> end(); ++itr){this -> _purge_data(itr -> second);}
    data -> clear(); 
}

std::vector<torch::Tensor*>* graph_t::add_content(std::map<std::string, torch::Tensor*>* inpt){
    std::vector<torch::Tensor*>* out = new std::vector<torch::Tensor*>(inpt -> size(), nullptr); 
    std::map<std::string, torch::Tensor*>::iterator itr = inpt -> begin();
    for (int t(0); t < inpt -> size(); ++t, ++itr){(*out)[t] = itr -> second;}
    return out; 
}

void graph_t::transfer_to_device(torch::TensorOptions* dev){
    c10::DeviceType dev_in = dev -> device().type(); 
    if (dev_in == c10::kCPU && this -> device == c10::kCPU){return;}
    int dev_ = (int)dev -> device().index(); 
    bool sm = (dev_in == c10::kCUDA && this -> device == c10::kCUDA); 
    sm *= this -> device_index[dev_]; 
    if (sm){return;}

    this -> _transfer_to_device(&this -> dev_data_graph[dev_] , this -> data_graph , dev);  
    this -> _transfer_to_device(&this -> dev_data_node[dev_]  , this -> data_node  , dev); 
    this -> _transfer_to_device(&this -> dev_data_edge[dev_]  , this -> data_edge  , dev); 
    this -> _transfer_to_device(&this -> dev_truth_graph[dev_], this -> truth_graph, dev); 
    this -> _transfer_to_device(&this -> dev_truth_node[dev_] , this -> truth_node , dev); 
    this -> _transfer_to_device(&this -> dev_truth_edge[dev_] , this -> truth_edge , dev); 
    this -> dev_edge_index[dev_] = new torch::Tensor(this -> edge_index -> to(dev -> device())); 

    this -> device_index[dev_] = true; 
    this -> device = dev_in; 
}

void graph_t::_transfer_to_device(std::vector<torch::Tensor*>** trgt, std::vector<torch::Tensor*>* src, torch::TensorOptions* dev){
    if (!src){return;}
    if (!(*trgt)){*trgt = new std::vector<torch::Tensor*>(src -> size(), nullptr);}
    else {return;}
    for (size_t x(0); x < src -> size(); ++x){
        torch::Tensor* ten = (*src)[x]; 
        torch::Tensor* ten_cu = new torch::Tensor(ten -> to(dev -> device())); 
        (**trgt)[x] = ten_cu; 
    }
}



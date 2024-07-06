#include <templates/graph_template.h>

torch::Tensor* graph_t::get_truth_graph(std::string name){
    name = "T-" + name;  
    if (!this -> truth_map_graph -> count(name)){return nullptr;}
    int x = (*this -> truth_map_graph)[name]; 
    return this -> truth_graph -> at(x); 
}

torch::Tensor* graph_t::get_truth_node(std::string name){
    name = "T-" + name; 
    if (!this -> truth_node -> size()){return nullptr;}
    if (!this -> truth_map_node -> count(name)){return nullptr;}
    int x = (*this -> truth_map_node)[name]; 
    return this -> truth_node -> at(x); 
}

torch::Tensor* graph_t::get_truth_edge(std::string name){
name = "T-" + name; 
    if (!this -> truth_map_edge -> count(name)){return nullptr;}
    int x = (*this -> truth_map_edge)[name]; 
    return this -> truth_edge -> at(x); 
}

torch::Tensor* graph_t::get_data_graph(std::string name){
    name = "D-" + name; 
    if (!this -> data_map_graph -> count(name)){return nullptr;}
    int x = (*this -> data_map_graph)[name]; 
    return this -> data_graph -> at(x); 
}

torch::Tensor* graph_t::get_data_node(std::string name){
    name = "D-" + name; 
    if (!this -> data_map_node -> count(name)){return nullptr;}
    int x = (*this -> data_map_node)[name]; 
    return this -> data_node -> at(x); 
}

torch::Tensor* graph_t::get_data_edge(std::string name){
    name = "D-" + name; 
    if (!this -> data_map_edge -> count(name)){return nullptr;}
    int x = (*this -> data_map_edge)[name]; 
    return this -> data_edge -> at(x); 
}

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
}

void graph_t::_purge_data(std::vector<torch::Tensor*>* data){
    if (!data){return;}
    for (size_t x(0); x < data -> size(); ++x){delete data -> at(x);}
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
    bool sm = (dev_in == c10::kCUDA && this -> device == c10::kCUDA); 
    sm *= this -> device_index == (int)dev -> device().index(); 
    if (sm){return;}

    this -> _transfer_to_device(this -> data_graph, dev);  
    this -> _transfer_to_device(this -> data_node, dev); 
    this -> _transfer_to_device(this -> data_edge, dev); 
    this -> _transfer_to_device(this -> truth_graph, dev); 
    this -> _transfer_to_device(this -> truth_node, dev); 
    this -> _transfer_to_device(this -> truth_edge, dev); 

    torch::Tensor* edge_indx_ = new torch::Tensor(this -> edge_index -> to(dev -> device())); 
    delete this -> edge_index; 
    this -> edge_index = edge_indx_; 

    this -> device_index = dev -> device().index(); 
    this -> device = dev_in; 
}

void graph_t::_transfer_to_device(std::vector<torch::Tensor*>* data, torch::TensorOptions* dev){
    if (!data){return;}
    for (size_t x(0); x < data -> size(); ++x){
        torch::Tensor* ten = data -> at(x); 
        torch::Tensor* ten_cu = new torch::Tensor(ten -> to(dev -> device())); 
        data -> at(x) = ten_cu; 
        delete ten; 
    }
}










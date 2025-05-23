#include <templates/graph_template.h>
#include <tools/tools.h>

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

    this -> _purge_data(this -> truth_graph); 
    this -> _purge_data(this -> truth_node); 
    this -> _purge_data(this -> truth_edge); 

    if (this -> data_graph ){delete this -> data_graph;  this -> data_graph  = nullptr;}
    if (this -> data_node  ){delete this -> data_node;   this -> data_node   = nullptr;} 
    if (this -> data_edge  ){delete this -> data_edge;   this -> data_edge   = nullptr;}
    if (this -> truth_graph){delete this -> truth_graph; this -> truth_graph = nullptr;}
    if (this -> truth_node ){delete this -> truth_node;  this -> truth_node  = nullptr;}
    if (this -> truth_edge ){delete this -> truth_edge;  this -> truth_edge  = nullptr;}
    if (this -> edge_index ){delete this -> edge_index;  this -> edge_index  = nullptr;}

    this -> dev_data_graph.clear(); 
    this -> dev_data_node.clear(); 
    this -> dev_data_edge.clear(); 

    this -> dev_truth_graph.clear(); 
    this -> dev_truth_node.clear(); 
    this -> dev_truth_edge.clear(); 

    this -> dev_edge_index.clear(); 
    this -> dev_event_weight.clear(); 
    this -> dev_batch_index.clear(); 

    if (this -> hash){delete this -> hash;}
    if (this -> graph_name){delete this -> graph_name;}
    if (!this -> is_owner){return;}
    delete this -> filename; 
}

torch::Tensor* graph_t::return_any(
    std::map<std::string, int>* loc, std::map<int, 
    std::vector<torch::Tensor>>* container, 
    std::string _name, int dev_
){
    if (!loc -> count(_name)){return nullptr;}
    int idx_ = (*loc)[_name];
    if (!container -> count(dev_)){return nullptr;} 
    if (int(container -> at(dev_).size()) <= idx_){return nullptr;}
    return &container -> at(dev_).at(idx_); 
}

torch::Tensor* graph_t::has_feature(graph_enum tp, std::string _name, int dev){
    std::map<std::string, int>* pos_ = nullptr; 
    std::map<int, std::vector<torch::Tensor>>* dev_mp = nullptr; 
    switch(tp){
        case graph_enum::data_edge   : pos_ = this -> data_map_edge  ; dev_mp = &this -> dev_data_edge  ; _name = "D-" + _name; break;  
        case graph_enum::data_node   : pos_ = this -> data_map_node  ; dev_mp = &this -> dev_data_node  ; _name = "D-" + _name; break;  
        case graph_enum::data_graph  : pos_ = this -> data_map_graph ; dev_mp = &this -> dev_data_graph ; _name = "D-" + _name; break;  
        case graph_enum::truth_edge  : pos_ = this -> truth_map_edge ; dev_mp = &this -> dev_truth_edge ; _name = "T-" + _name; break;  
        case graph_enum::truth_node  : pos_ = this -> truth_map_node ; dev_mp = &this -> dev_truth_node ; _name = "T-" + _name; break;  
        case graph_enum::truth_graph : pos_ = this -> truth_map_graph; dev_mp = &this -> dev_truth_graph; _name = "T-" + _name; break;  
        case graph_enum::edge_index  : return (this -> dev_edge_index.count(dev)) ? &this -> dev_edge_index[dev] : nullptr; 
        case graph_enum::weight      : return (this -> dev_event_weight.count(dev)) ? &this -> dev_event_weight[dev] : nullptr; 
        case graph_enum::batch_index : return (this -> dev_batch_index.count(dev)) ? &this -> dev_batch_index[dev] : nullptr; 
        case graph_enum::batch_events: return (this -> dev_batched_events.count(dev)) ? &this -> dev_batched_events[dev] : nullptr; 
        default: return nullptr; 
    }
    return this -> return_any(pos_, dev_mp, _name, dev); 
}

void graph_t::_purge_data(std::map<int, torch::Tensor*>* data){
    if (!data){return;}
    std::map<int, torch::Tensor*>::iterator itr = data -> begin();
    for (; itr != data -> end(); ++itr){delete itr -> second;}
    data -> clear(); 
}

void graph_t::_purge_data(std::vector<torch::Tensor*>* data){
    if (!data){return;}
    for (size_t x(0); x < data -> size(); ++x){
        if (!(*data)[x]){continue;}
        delete (*data)[x]; (*data)[x] = nullptr;
    }
}

void graph_t::_purge_data(std::map<int, std::vector<torch::Tensor*>*>* data){
    if (!data){return;}
    std::map<int, std::vector<torch::Tensor*>*>::iterator itr = data -> begin();
    for (; itr != data -> end(); ++itr){
        if (!itr -> second){continue;}
        this -> _purge_data(itr -> second);
        delete itr -> second; itr -> second = nullptr; 
    }
    data -> clear(); 
}

std::vector<torch::Tensor*>* graph_t::add_content(std::map<std::string, torch::Tensor*>* inpt){
    std::vector<torch::Tensor*>* out = new std::vector<torch::Tensor*>(inpt -> size(), nullptr); 
    std::map<std::string, torch::Tensor*>::iterator itr = inpt -> begin();
    for (size_t t(0); t < inpt -> size(); ++t, ++itr){(*out)[t] = itr -> second;}
    return out; 
}

void graph_t::transfer_to_device(torch::TensorOptions* dev){
    c10::DeviceType dev_in = dev -> device().type(); 
    if (dev_in == c10::kCPU){return;}
    int dev_ = (int)dev -> device().index(); 
    bool sm = (dev_in == c10::kCUDA && this -> device == c10::kCUDA); 
    sm = sm && this -> device_index[dev_]; 
    if (sm){return;}
    this -> device_index[dev_] = true; 
    std::unique_lock<std::mutex> lk(this -> mut); 
    this -> _transfer_to_device(&this -> dev_data_graph[dev_] , this -> data_graph , dev);  
    this -> _transfer_to_device(&this -> dev_data_node[dev_]  , this -> data_node  , dev); 
    this -> _transfer_to_device(&this -> dev_data_edge[dev_]  , this -> data_edge  , dev); 
    this -> _transfer_to_device(&this -> dev_truth_graph[dev_], this -> truth_graph, dev); 
    this -> _transfer_to_device(&this -> dev_truth_node[dev_] , this -> truth_node , dev); 
    this -> _transfer_to_device(&this -> dev_truth_edge[dev_] , this -> truth_edge , dev); 

    this -> batched_events = std::vector<long>({0}); 
    std::vector<long> bc(this -> num_nodes, 0);
    std::vector<double> evw = {this -> event_weight}; 
    torch::TensorOptions op = torch::TensorOptions(torch::kCPU); 
    
    torch::Tensor dt = build_tensor(&evw, torch::kDouble, double(), &op);
    torch::Tensor bx = build_tensor(&bc , torch::kLong, long(), &op); 
    torch::Tensor bi = build_tensor(&this -> batched_events, torch::kLong, long(), &op); 

    this -> dev_event_weight[dev_]   = dt.clone().to(dev -> device(), true); 
    this -> dev_batch_index[dev_]    = bx.clone().to(dev -> device(), true); 
    this -> dev_batched_events[dev_] = bi.clone().to(dev -> device(), true);
    this -> dev_edge_index[dev_]     = this -> edge_index -> to(dev -> device(), true); 
    torch::cuda::synchronize(dev_); 
    this -> device = dev_in;
    lk.unlock(); 
}

void graph_t::_transfer_to_device(std::vector<torch::Tensor>* trgt, std::vector<torch::Tensor*>* src, torch::TensorOptions* dev){
    if (!src || trgt -> size()){return;}
    trgt -> reserve(src -> size()); 
    for (size_t x(0); x < src -> size(); ++x){trgt -> push_back((*src)[x] -> to(dev -> device(), true));}
}

void graph_t::meta_serialize(std::map<std::string, int>* data, std::string* out){
    if (!data -> size()){*out = "NULL"; return;}
    std::map<std::string, int>::iterator it = data -> begin();
    for (; it != data -> end(); ++it){*out += it -> first + "|" + std::to_string(it -> second) + "%";}
}

void graph_t::meta_serialize(std::vector<torch::Tensor*>* data, std::string* out){
    if (!data -> size()){*out = "NULL"; return;}
    std::vector<torch::Tensor> data_; 
    data_.reserve(data -> size()); 
    for (torch::Tensor* t : *data){data_.push_back(*t);}
    std::vector<char> chars = torch::pickle_save(data_); 

    tools tl = tools(); 
    *out = std::string(chars.begin(), chars.end()); 
    *out = tl.encode64((const unsigned char*)out -> c_str(), out -> size()); 
}

void graph_t::meta_serialize(torch::Tensor* data, std::string* out){
    std::vector<char> chars = torch::pickle_save(*data); 
    if (!chars.size()){*out = "NULL"; return;}

    tools tl = tools();
    *out = std::string(chars.begin(), chars.end()); 
    *out = tl.encode64((const unsigned char*)out -> c_str(), out -> size()); 
}

void graph_t::serialize(graph_hdf5* m_hdf5){
    m_hdf5 -> hash         = this -> hash -> c_str();
    m_hdf5 -> filename     = this -> filename -> c_str();
    m_hdf5 -> num_nodes    = this -> num_nodes;
    m_hdf5 -> event_index  = this -> event_index;
    m_hdf5 -> event_weight = this -> event_weight; 

    this -> meta_serialize(this -> edge_index, &m_hdf5 -> edge_index); 

    this -> meta_serialize(this -> data_map_graph , &m_hdf5 -> data_map_graph );
    this -> meta_serialize(this -> data_map_node  , &m_hdf5 -> data_map_node  );
    this -> meta_serialize(this -> data_map_edge  , &m_hdf5 -> data_map_edge  );

    this -> meta_serialize(this -> truth_map_graph, &m_hdf5 -> truth_map_graph);
    this -> meta_serialize(this -> truth_map_node , &m_hdf5 -> truth_map_node );
    this -> meta_serialize(this -> truth_map_edge , &m_hdf5 -> truth_map_edge );

    this -> meta_serialize(this -> data_graph , &m_hdf5 -> data_graph );
    this -> meta_serialize(this -> data_node  , &m_hdf5 -> data_node  );
    this -> meta_serialize(this -> data_edge  , &m_hdf5 -> data_edge  );

    this -> meta_serialize(this -> truth_graph, &m_hdf5 -> truth_graph);
    this -> meta_serialize(this -> truth_node , &m_hdf5 -> truth_node );
    this -> meta_serialize(this -> truth_edge , &m_hdf5 -> truth_edge );
}


void graph_t::meta_deserialize(std::map<std::string, int>* data, std::string* out){
    if ((*out) == "NULL"){return;}

    tools t = tools(); 
    std::vector<std::string> line = t.split(*out, "%"); 
    for (size_t x(0); x < line.size()-1; ++x){
        std::vector<std::string> c = t.split(line[x], "|"); 
        (*data)[c[0]] = std::stoi(c[1]); 
    }
}

void graph_t::meta_deserialize(std::vector<torch::Tensor*>* data, std::string* out){
    if ((*out) == "NULL"){return;}

    tools tl = tools();
    std::string tmp = tl.decode64(out); 
    std::vector<char> pkl(tmp.begin(), tmp.end()); 
    std::vector<torch::Tensor> datav = torch::pickle_load(pkl).toTensorVector();  
    data -> assign(datav.size(), nullptr); 
    for (size_t x(0); x < datav.size(); ++x){(*data)[x] = new torch::Tensor(datav[x]);}
    datav.clear(); 
}

torch::Tensor* graph_t::meta_deserialize(std::string* out){
    if ((*out) == "NULL"){return nullptr;}

    tools tl = tools();
    std::string tmp = tl.decode64(out);  
    std::vector<char> pkl(tmp.begin(), tmp.end()); 
    return new torch::Tensor(torch::pickle_load(pkl).toTensor());  
}

void graph_t::deserialize(graph_hdf5* m_hdf5){
    this -> hash         = new std::string(m_hdf5 -> hash);
    this -> filename     = new std::string(m_hdf5 -> filename);
    this -> num_nodes    = m_hdf5 -> num_nodes;
    this -> preselection = !this -> num_nodes; 
    this -> event_index  = m_hdf5 -> event_index;
    this -> event_weight = m_hdf5 -> event_weight;

    this -> data_map_graph  = new std::map<std::string, int>();  
    this -> data_map_node   = new std::map<std::string, int>();       
    this -> data_map_edge   = new std::map<std::string, int>();      

    this -> meta_deserialize(this -> data_map_graph , &m_hdf5 -> data_map_graph );   
    this -> meta_deserialize(this -> data_map_node  , &m_hdf5 -> data_map_node  );   
    this -> meta_deserialize(this -> data_map_edge  , &m_hdf5 -> data_map_edge  );   

    this -> truth_map_graph = new std::map<std::string, int>();  
    this -> truth_map_node  = new std::map<std::string, int>();       
    this -> truth_map_edge  = new std::map<std::string, int>();       

    this -> meta_deserialize(this -> truth_map_graph, &m_hdf5 -> truth_map_graph);   
    this -> meta_deserialize(this -> truth_map_node , &m_hdf5 -> truth_map_node );   
    this -> meta_deserialize(this -> truth_map_edge , &m_hdf5 -> truth_map_edge );   

    this -> edge_index = this -> meta_deserialize(&m_hdf5 -> edge_index); 
    this -> data_graph = new std::vector<torch::Tensor*>();  
    this -> data_node  = new std::vector<torch::Tensor*>();  
    this -> data_edge  = new std::vector<torch::Tensor*>();  

    this -> meta_deserialize(this -> data_graph, &m_hdf5 -> data_graph );       
    this -> meta_deserialize(this -> data_node , &m_hdf5 -> data_node  );       
    this -> meta_deserialize(this -> data_edge , &m_hdf5 -> data_edge  );       

    this -> truth_graph = new std::vector<torch::Tensor*>();  
    this -> truth_node  = new std::vector<torch::Tensor*>();  
    this -> truth_edge  = new std::vector<torch::Tensor*>();  

    this -> meta_deserialize(this -> truth_graph, &m_hdf5 -> truth_graph);       
    this -> meta_deserialize(this -> truth_node , &m_hdf5 -> truth_node );       
    this -> meta_deserialize(this -> truth_edge , &m_hdf5 -> truth_edge );       
    this -> is_owner = true; 
}


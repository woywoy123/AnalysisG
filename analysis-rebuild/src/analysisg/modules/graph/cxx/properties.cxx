#include <templates/graph_template.h>

void graph_template::set_name(std::string* name, graph_template* ev){
    ev -> data.name = *name; 
}

void graph_template::get_hash(std::string* val, graph_template* ev){
    *val = ev -> data.hash; 
}

void graph_template::get_tree(std::string* name, graph_template* ev){
    *name = ev -> data.name; 
}

void graph_template::get_index(long* inpt, graph_template* ev){
    *inpt = ev -> data.index; 
}

void graph_template::set_device(std::string* dev, graph_template* gr){
    if (!gr -> op){gr -> op = new torch::TensorOptions();}
    if (*dev == "cuda"){gr -> op -> device(torch::kCUDA, 0);} 
    else {gr -> op -> device(torch::kCPU);}
    gr -> m_device = *dev;  
}

void graph_template::get_device(std::string* dev, graph_template* gr){
    *dev = gr -> m_device;  
    if (!gr -> op){gr -> set_device(dev, gr);}
}

void graph_template::add_graph_feature(bool data, std::string name){
    if (this -> graph_fx.count(name)){return;}
    std::cout << "here " << std::endl;
    this -> graph_fx["G-" + name] = this -> to_tensor(std::vector<bool>{data}, torch::kBool); 
}

void graph_template::add_graph_feature(std::vector<bool> _data, std::string name){
    if (this -> graph_fx.count(name)){return;}
    this -> graph_fx["G-" + name] = this -> to_tensor(_data, torch::kBool); 
}

void graph_template::add_graph_feature(float data, std::string name){
    if (this -> graph_fx.count(name)){return;}
    this -> graph_fx["G-" + name] = this -> to_tensor(std::vector<float>{data}, torch::kFloat); 
}

void graph_template::add_graph_feature(std::vector<float> _data, std::string name){
    if (this -> graph_fx.count(name)){return;}
    this -> graph_fx["G-" + name] = this -> to_tensor(_data, torch::kFloat); 
}

void graph_template::add_graph_feature(double data, std::string name){
    if (this -> graph_fx.count(name)){return;}
    this -> graph_fx["G-" + name] = this -> to_tensor(std::vector<double>{data}, torch::kDouble); 
}

void graph_template::add_graph_feature(std::vector<double> _data, std::string name){
    if (this -> graph_fx.count(name)){return;}
    this -> graph_fx["G-" + name] = this -> to_tensor(_data, torch::kDouble); 
}

void graph_template::add_graph_feature(long data, std::string name){
    if (this -> graph_fx.count(name)){return;}
    this -> graph_fx["G-" + name] = this -> to_tensor(std::vector<long>{data}, torch::kLong); 
}

void graph_template::add_graph_feature(std::vector<long> _data, std::string name){
    if (this -> graph_fx.count(name)){return;}
    this -> graph_fx["G-" + name] = this -> to_tensor(_data, torch::kLong); 
}

void graph_template::add_graph_feature(int data, std::string name){
    if (this -> graph_fx.count(name)){return;}
    this -> graph_fx["G-" + name] = this -> to_tensor(std::vector<int>{data}, torch::kInt); 
}

void graph_template::add_graph_feature(std::vector<int> _data, std::string name){
    if (this -> graph_fx.count(name)){return;}
    this -> graph_fx["G-" + name] = this -> to_tensor(_data, torch::kInt); 
}

void graph_template::add_node_feature(bool data, std::string name){
    if (this -> node_fx.count(name)){return;}
    this -> node_fx["N-" + name] = this -> to_tensor(std::vector<bool>{data}, torch::kBool); 
}

void graph_template::add_node_feature(std::vector<bool> _data, std::string name){
    if (this -> node_fx.count(name)){return;}
    this -> node_fx["N-" + name] = this -> to_tensor(_data, torch::kBool); 
}

void graph_template::add_node_feature(float data, std::string name){
    if (this -> node_fx.count(name)){return;}
    this -> node_fx["N-" + name] = this -> to_tensor(std::vector<float>{data}, torch::kFloat); 
}

void graph_template::add_node_feature(std::vector<float> _data, std::string name){
    if (this -> node_fx.count(name)){return;}
    this -> node_fx["N-" + name] = this -> to_tensor(_data, torch::kFloat); 
}

void graph_template::add_node_feature(double data, std::string name){
    if (this -> node_fx.count(name)){return;}
    this -> node_fx["N-" + name] = this -> to_tensor(std::vector<double>{data}, torch::kDouble); 
}

void graph_template::add_node_feature(std::vector<double> _data, std::string name){
    if (this -> node_fx.count(name)){return;}
    this -> node_fx["N-" + name] = this -> to_tensor(_data, torch::kDouble); 
}

void graph_template::add_node_feature(long data, std::string name){
    if (this -> node_fx.count(name)){return;}
    this -> node_fx["N-" + name] = this -> to_tensor(std::vector<long>{data}, torch::kLong); 
}

void graph_template::add_node_feature(std::vector<long> _data, std::string name){
    if (this -> node_fx.count(name)){return;}
    this -> node_fx["N-" + name] = this -> to_tensor(_data, torch::kLong); 
}

void graph_template::add_node_feature(int data, std::string name){
    if (this -> node_fx.count(name)){return;}
    this -> node_fx["N-" + name] = this -> to_tensor(std::vector<int>{data}, torch::kInt); 
}

void graph_template::add_node_feature(std::vector<int> _data, std::string name){
    if (this -> node_fx.count(name)){return;}
    this -> node_fx["N-" + name] = this -> to_tensor(_data, torch::kInt); 
}

void graph_template::add_edge_feature(bool data, std::string name){
    if (this -> edge_fx.count(name)){return;}
    this -> edge_fx["E-" + name] = this -> to_tensor(std::vector<bool>{data}, torch::kBool); 
}

void graph_template::add_edge_feature(std::vector<bool> _data, std::string name){
    if (this -> edge_fx.count(name)){return;}
    this -> edge_fx["E-" + name] = this -> to_tensor(_data, torch::kBool); 
}

void graph_template::add_edge_feature(float data, std::string name){
    if (this -> edge_fx.count(name)){return;}
    this -> edge_fx["E-" + name] = this -> to_tensor(std::vector<float>{data}, torch::kFloat); 
}

void graph_template::add_edge_feature(std::vector<float> _data, std::string name){
    if (this -> edge_fx.count(name)){return;}
    this -> edge_fx["E-" + name] = this -> to_tensor(_data, torch::kFloat); 
}

void graph_template::add_edge_feature(double data, std::string name){
    if (this -> edge_fx.count(name)){return;}
    this -> edge_fx["E-" + name] = this -> to_tensor(std::vector<double>{data}, torch::kDouble); 
}

void graph_template::add_edge_feature(std::vector<double> _data, std::string name){
    if (this -> edge_fx.count(name)){return;}
    this -> edge_fx["E-" + name] = this -> to_tensor(_data, torch::kDouble); 
}

void graph_template::add_edge_feature(long data, std::string name){
    if (this -> edge_fx.count(name)){return;}
    this -> edge_fx["E-" + name] = this -> to_tensor(std::vector<long>{data}, torch::kLong); 
}

void graph_template::add_edge_feature(std::vector<long> _data, std::string name){
    if (this -> edge_fx.count(name)){return;}
    this -> edge_fx["E-" + name] = this -> to_tensor(_data, torch::kLong); 
}

void graph_template::add_edge_feature(int data, std::string name){
    if (this -> edge_fx.count(name)){return;}
    this -> edge_fx["E-" + name] = this -> to_tensor(std::vector<int>{data}, torch::kInt); 
}

void graph_template::add_edge_feature(std::vector<int> _data, std::string name){
    if (this -> edge_fx.count(name)){return;}
    this -> edge_fx["E-" + name] = this -> to_tensor(_data, torch::kInt); 
}




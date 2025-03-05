#include <templates/graph_template.h>

void graph_template::set_name(std::string* name, graph_template* ev){
    ev -> data.name = *name; 
}

void graph_template::get_hash(std::string* val, graph_template* ev){
    *val = ev -> data.hash; 
}

void graph_template::get_tree(std::string* name, graph_template* ev){
    *name = ev -> data.tree; 
}

void graph_template::get_index(long* inpt, graph_template* ev){
    *inpt = ev -> data.index; 
}

void graph_template::get_weight(double* inpt, graph_template* ev){
    *inpt = ev -> data.weight; 
}

void graph_template::set_preselection(bool* inpt, graph_template* ev){
    ev -> m_preselection = *inpt; 
}

void graph_template::get_preselection(bool* inpt, graph_template* ev){
    *inpt = ev -> m_preselection; 
}


void graph_template::add_graph_feature(bool data, std::string name){
    this -> graph_fx[name] = this -> to_tensor(std::vector<bool>{data}, torch::kBool, bool()); 
}

void graph_template::add_graph_feature(std::vector<bool> _data, std::string name){
    this -> graph_fx[name] = this -> to_tensor(_data, torch::kBool, bool()); 
}

void graph_template::add_graph_feature(float data, std::string name){
    this -> graph_fx[name] = this -> to_tensor(std::vector<float>{data}, torch::kFloat, float()); 
}

void graph_template::add_graph_feature(std::vector<float> _data, std::string name){
    this -> graph_fx[name] = this -> to_tensor(_data, torch::kFloat, float()); 
}

void graph_template::add_graph_feature(double data, std::string name){
    this -> graph_fx[name] = this -> to_tensor(std::vector<double>{data}, torch::kDouble, double()); 
}

void graph_template::add_graph_feature(std::vector<double> _data, std::string name){
    this -> graph_fx[name] = this -> to_tensor(_data, torch::kDouble, double()); 
}

void graph_template::add_graph_feature(long data, std::string name){
    this -> graph_fx[name] = this -> to_tensor(std::vector<long>{data}, torch::kLong, long()); 
}

void graph_template::add_graph_feature(std::vector<long> _data, std::string name){
    this -> graph_fx[name] = this -> to_tensor(_data, torch::kLong, long()); 
}

void graph_template::add_graph_feature(int data, std::string name){
    this -> graph_fx[name] = this -> to_tensor(std::vector<int>{data}, torch::kInt, int()); 
}

void graph_template::add_graph_feature(std::vector<int> _data, std::string name){
    this -> graph_fx[name] = this -> to_tensor(_data, torch::kInt, int()); 
}

void graph_template::add_graph_feature(std::vector<std::vector<int>> _data, std::string name){
    this -> graph_fx[name] = this -> to_tensor(_data, torch::kInt, int()); 
}



void graph_template::add_node_feature(bool data, std::string name){
    this -> node_fx[name] = this -> to_tensor(std::vector<bool>{data}, torch::kBool, bool()); 
}

void graph_template::add_node_feature(std::vector<bool> _data, std::string name){
    this -> node_fx[name] = this -> to_tensor(_data, torch::kBool, bool()); 
}

void graph_template::add_node_feature(float data, std::string name){
    this -> node_fx[name] = this -> to_tensor(std::vector<float>{data}, torch::kFloat, float()); 
}

void graph_template::add_node_feature(std::vector<float> _data, std::string name){
    this -> node_fx[name] = this -> to_tensor(_data, torch::kFloat, float()); 
}

void graph_template::add_node_feature(double data, std::string name){
    this -> node_fx[name] = this -> to_tensor(std::vector<double>{data}, torch::kDouble, double()); 
}

void graph_template::add_node_feature(std::vector<double> _data, std::string name){
    this -> node_fx[name] = this -> to_tensor(_data, torch::kDouble, double()); 
}

void graph_template::add_node_feature(long data, std::string name){
    this -> node_fx[name] = this -> to_tensor(std::vector<long>{data}, torch::kLong, long()); 
}

void graph_template::add_node_feature(std::vector<long> _data, std::string name){
    this -> node_fx[name] = this -> to_tensor(_data, torch::kLong, long()); 
}

void graph_template::add_node_feature(int data, std::string name){
    this -> node_fx[name] = this -> to_tensor(std::vector<int>{data}, torch::kInt, int()); 
}

void graph_template::add_node_feature(std::vector<int> _data, std::string name){
    this -> node_fx[name] = this -> to_tensor(_data, torch::kInt, int()); 
}

void graph_template::add_node_feature(std::vector<std::vector<int>> _data, std::string name){
    this -> node_fx[name] = this -> to_tensor(_data, torch::kInt, int()); 
}


void graph_template::add_edge_feature(bool data, std::string name){
    this -> edge_fx[name] = this -> to_tensor(std::vector<bool>{data}, torch::kBool, bool()); 
}

void graph_template::add_edge_feature(std::vector<bool> _data, std::string name){
    this -> edge_fx[name] = this -> to_tensor(_data, torch::kBool, bool()); 
}

void graph_template::add_edge_feature(float data, std::string name){
    this -> edge_fx[name] = this -> to_tensor(std::vector<float>{data}, torch::kFloat, float()); 
}

void graph_template::add_edge_feature(std::vector<float> _data, std::string name){
    this -> edge_fx[name] = this -> to_tensor(_data, torch::kFloat, float()); 
}

void graph_template::add_edge_feature(double data, std::string name){
    this -> edge_fx[name] = this -> to_tensor(std::vector<double>{data}, torch::kDouble, double()); 
}

void graph_template::add_edge_feature(std::vector<double> _data, std::string name){
    this -> edge_fx[name] = this -> to_tensor(_data, torch::kDouble, double()); 
}

void graph_template::add_edge_feature(long data, std::string name){
    this -> edge_fx[name] = this -> to_tensor(std::vector<long>{data}, torch::kLong, long()); 
}

void graph_template::add_edge_feature(std::vector<long> _data, std::string name){
    this -> edge_fx[name] = this -> to_tensor(_data, torch::kLong, long()); 
}

void graph_template::add_edge_feature(int data, std::string name){
    this -> edge_fx[name] = this -> to_tensor(std::vector<int>{data}, torch::kInt, int()); 
}

void graph_template::add_edge_feature(std::vector<int> _data, std::string name){
    this -> edge_fx[name] = this -> to_tensor(_data, torch::kInt, int()); 
}

void graph_template::add_edge_feature(std::vector<std::vector<int>> _data, std::string name){
    this -> edge_fx[name] = this -> to_tensor(_data, torch::kInt, int()); 
}


void graph_template::build_export(
        std::map<std::string, torch::Tensor*>* _truth_t, std::map<std::string, int>* _truth_i,
        std::map<std::string, torch::Tensor*>* _data_t , std::map<std::string, int>*  _data_i,
        std::map<std::string, torch::Tensor>* _fx
){
    std::map<std::string, torch::Tensor>::iterator itr = _fx -> begin();
    for (; itr != _fx -> end(); ++itr){

        bool tru = itr -> first.substr(0, 2) == "T-"; 
        std::map<std::string, torch::Tensor*>* mt = nullptr; 
        std::map<std::string, int>*            mx = nullptr; 

        if (tru){mx = _truth_i; mt = _truth_t; }
        else {mx = _data_i; mt = _data_t; }
        (*mx)[itr -> first] = mx -> size();
        (*mt)[itr -> first] = new torch::Tensor(itr -> second);
    }
}






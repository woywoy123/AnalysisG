#include <chrono>
#include <templates/model_template.h>
#include <thread>

model_template::model_template(){
    // input features
    this -> i_graph.set_setter(this -> set_input_features);
    this -> i_graph.set_object(&this -> m_i_graph); 

    this -> i_node.set_setter(this -> set_input_features);
    this -> i_node.set_object(&this -> m_i_node); 

    this -> i_edge.set_setter(this -> set_input_features);
    this -> i_edge.set_object(&this -> m_i_edge);
   
    // output features 
    this -> o_graph.set_setter(this -> set_output_features); 
    this -> o_graph.set_object(&this -> m_o_graph); 

    this -> o_node.set_setter(this -> set_output_features); 
    this -> o_node.set_object(&this -> m_o_node); 

    this -> o_edge.set_setter(this -> set_output_features); 
    this -> o_edge.set_object(&this -> m_o_edge); 

    this -> device.set_setter(this -> set_device); 
    this -> device.set_object(this); 

    this -> m_loss = new lossfx(); 
}

model_template* model_template::clone(){return new model_template();}

void model_template::register_module(torch::nn::Sequential* data){
    if (this -> m_option){(*data) -> to(this -> m_option -> device());}
    this -> m_data.push_back(data);
}

void model_template::register_module(torch::nn::Sequential* data, mlp_init method){
    this -> m_loss -> weight_init(data, method); 
    this -> m_data.push_back(data);
}

void model_template::forward(graph_t* data){}

model_template::~model_template(){
    for (size_t x(0); x < this -> m_data.size(); ++x){delete this -> m_data[x];}
    this -> flush_outputs(); 
    this -> m_data.clear(); 
    delete this -> m_loss; 
}

torch::Tensor* model_template::assign_features(std::string inpt, graph_enum type, graph_t* data){
    torch::Tensor* tn = nullptr; 
    switch (type){
        case graph_enum::data_graph:  tn = data -> get_data_graph(inpt, this);  this -> m_i_graph[inpt] = tn; break;  
        case graph_enum::data_node:   tn = data -> get_data_node(inpt, this);   this -> m_i_node[inpt]  = tn; break;  
        case graph_enum::data_edge:   tn = data -> get_data_edge(inpt, this);   this -> m_i_edge[inpt]  = tn; break;  
        case graph_enum::truth_graph: tn = data -> get_truth_graph(inpt, this); std::get<0>(this -> m_o_graph[inpt]) = tn; break;  
        case graph_enum::truth_node:  tn = data -> get_truth_node(inpt, this);  std::get<0>(this -> m_o_node[inpt])  = tn; break;  
        case graph_enum::truth_edge:  tn = data -> get_truth_edge(inpt, this);  std::get<0>(this -> m_o_edge[inpt])  = tn; break;  
        default: return tn; 
    }
    return tn; 
}

void model_template::forward(graph_t* data, bool train){
    if (data -> in_use == 0){
        while (data -> in_use == 0){std::this_thread::sleep_for(std::chrono::milliseconds(1));}
    }
    data -> transfer_to_device(this -> m_option); 
    this -> flush_outputs(); 

    graph_enum mode; 
    std::map<std::string, torch::Tensor*>::iterator itr;
    std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>::iterator itx; 

    this -> edge_index = data -> get_edge_index(this); 

    mode = graph_enum::data_graph; 
    itr = this -> m_i_graph.begin(); 
    for (; itr != this -> m_i_graph.end(); ++itr){this -> assign_features(itr -> first, mode, data);}

    mode = graph_enum::data_node; 
    itr = this -> m_i_node.begin(); 
    for (; itr != this -> m_i_node.end(); ++itr){this -> assign_features(itr -> first, mode, data);}

    mode = graph_enum::data_edge; 
    itr = this -> m_i_edge.begin(); 
    for (; itr != this -> m_i_edge.end(); ++itr){this -> assign_features(itr -> first, mode, data);}

    mode = graph_enum::truth_graph; 
    itx = this -> m_o_graph.begin(); 
    for (; itx != this -> m_o_graph.end(); ++itx){this -> assign_features(itx -> first, mode, data);}
   
    mode = graph_enum::truth_node; 
    itx = this -> m_o_node.begin(); 
    for (; itx != this -> m_o_node.end(); ++itx){this -> assign_features(itx -> first, mode, data);}

    mode = graph_enum::truth_edge; 
    itx = this -> m_o_edge.begin(); 
    for (; itx != this -> m_o_edge.end(); ++itx){this -> assign_features(itx -> first, mode, data);}

    this -> forward(data); 
    this -> train_sequence(train);
}


void model_template::set_input_features(std::vector<std::string>* inpt, std::map<std::string, torch::Tensor*>* in_fx){
    for (size_t x(0); x < inpt -> size(); ++x){(*in_fx)[inpt -> at(x)] = nullptr;}
}

void model_template::evaluation_mode(bool mode){
    for (size_t x(0); x < this -> m_data.size(); ++x){
        if (mode){(*this -> m_data[x]) -> eval();}
        else {(*this -> m_data[x]) -> train(true);}
    }
}

void model_template::prediction_graph_feature(std::string key, torch::Tensor pred){
    if (!this -> m_o_graph.count(key)){this -> warning("Graph Output Feature: " + key + " not found.");}
    else {this -> m_p_graph[key] = new torch::Tensor(pred);}
}

void model_template::prediction_node_feature(std::string key, torch::Tensor pred){
    if (!this -> m_o_node.count(key)){this -> warning("Node Output Feature: " + key + " not found.");}
    else {this -> m_p_node[key] = new torch::Tensor(pred);}
}

void model_template::prediction_edge_feature(std::string key, torch::Tensor pred){
    if (!this -> m_o_edge.count(key)){this -> warning("Edge Output Feature: " + key + " not found.");}
    else {this -> m_p_edge[key] = new torch::Tensor(pred);}
}

void model_template::prediction_extra(std::string key, torch::Tensor pred){
    this -> m_p_undef[key] = new torch::Tensor(pred); 
}

void model_template::flush_outputs(){
    auto lambda = [](std::map<std::string, torch::Tensor*>* data){
        std::map<std::string, torch::Tensor*>::iterator itr = data -> begin(); 
        for (; itr != data -> end(); ++itr){
            if (!(*data)[itr -> first]){continue;}
            delete itr -> second; 
            (*data)[itr -> first] = nullptr; 
        }
    }; 

    lambda(&this -> m_p_graph); 
    lambda(&this -> m_p_node); 
    lambda(&this -> m_p_edge); 
    lambda(&this -> m_p_undef); 
}



#include <templates/model_template.h>

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
        case graph_enum::data_graph:  tn = data -> get_data_graph(inpt);  this -> m_i_graph[inpt] = tn; break;  
        case graph_enum::data_node:   tn = data -> get_data_node(inpt);   this -> m_i_node[inpt]  = tn; break;  
        case graph_enum::data_edge:   tn = data -> get_data_edge(inpt);   this -> m_i_edge[inpt]  = tn; break;  
        case graph_enum::truth_graph: tn = data -> get_truth_graph(inpt); std::get<0>(this -> m_o_graph[inpt]) = tn; break;  
        case graph_enum::truth_node:  tn = data -> get_truth_node(inpt);  std::get<0>(this -> m_o_node[inpt])  = tn; break;  
        case graph_enum::truth_edge:  tn = data -> get_truth_edge(inpt);  std::get<0>(this -> m_o_edge[inpt])  = tn; break;  
        default: return nullptr; 
    }
    return tn; 
}

void model_template::forward(graph_t* data, bool train){
    data -> transfer_to_device(this -> m_option); 
    this -> flush_outputs(); 

    std::map<std::string, torch::Tensor*>::iterator itr;
    for (itr = this -> m_i_graph.begin(); itr != this -> m_i_graph.end(); ++itr){
        this -> assign_features(itr -> first, graph_enum::data_graph, data);
    }

    for (itr = this -> m_i_node.begin(); itr != this -> m_i_node.end(); ++itr){
        this -> assign_features(itr -> first, graph_enum::data_node, data);
    }

    for (itr = this -> m_i_edge.begin(); itr != this -> m_i_edge.end(); ++itr){
        this -> assign_features(itr -> first, graph_enum::data_edge, data);
    }

    std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>::iterator itx; 
    for (itx = this -> m_o_graph.begin(); itx != this -> m_o_graph.end(); ++itx){
        this -> assign_features(itx -> first, graph_enum::truth_graph, data);
    }
    
    for (itx = this -> m_o_node.begin(); itx != this -> m_o_node.end(); ++itx){
        this -> assign_features(itx -> first, graph_enum::truth_node, data);
    }
 
    for (itx = this -> m_o_edge.begin(); itx != this -> m_o_edge.end(); ++itx){
        this -> assign_features(itx -> first, graph_enum::truth_edge, data);
    }
    this -> edge_index = data -> edge_index; 

    this -> forward(data); 
    this -> train_sequence(train);
}


void model_template::set_input_features(std::vector<std::string>* inpt, std::map<std::string, torch::Tensor*>* in_fx){
    for (size_t x(0); x < inpt -> size(); ++x){
        (*in_fx)[inpt -> at(x)] = nullptr;
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

void model_template::flush_outputs(){
    std::map<std::string, torch::Tensor*>::iterator itr; 
    for (itr = this -> m_p_graph.begin(); itr != this -> m_p_graph.end(); ++itr){
        if (!itr -> second){continue;}
        delete itr -> second; 
        this -> m_p_graph[itr -> first] = nullptr; 
    }

    for (itr = this -> m_p_node.begin(); itr != this -> m_p_node.end(); ++itr){
        if (!itr -> second){continue;}
        delete itr -> second; 
        this -> m_p_node[itr -> first] = nullptr; 
    }

    for (itr = this -> m_p_edge.begin(); itr != this -> m_p_edge.end(); ++itr){
        if (!itr -> second){continue;}
        delete itr -> second; 
        this -> m_p_edge[itr -> first] = nullptr; 
    }
}



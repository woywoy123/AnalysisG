#include <templates/model_template.h>

void model_template::check_features(graph_t* data){
    this -> flush_outputs(); 
    data -> transfer_to_device(this -> m_option); 

    std::map<std::string, torch::Tensor*>::iterator itr; 
    for (itr = this -> m_i_graph.begin(); itr != this -> m_i_graph.end(); ++itr){
        std::string key = "Input Graph Feature: " + itr -> first; 
        torch::Tensor* ten = this -> assign_features(itr -> first, graph_enum::data_graph, data); 
        if (!ten){this -> warning(key + " not found in graph.");}
        else {this -> success(key + " found in graph");}
    }

    for (itr = this -> m_i_node.begin(); itr != this -> m_i_node.end(); ++itr){
        std::string key = "Input Node Feature: " + itr -> first; 
        torch::Tensor* ten = this -> assign_features(itr -> first, graph_enum::data_node, data); 
        if (!ten){this -> warning(key + " not found in graph.");}
        else {this -> success(key + " found in graph.");}
    }

    for (itr = this -> m_i_edge.begin(); itr != this -> m_i_edge.end(); ++itr){
        std::string key = "Input Edge Feature: " + itr -> first; 
        torch::Tensor* ten = this -> assign_features(itr -> first, graph_enum::data_edge, data); 
        if (!ten){this -> warning(key + " not found in graph");}
        else {this -> success(key + " found in graph");}
    }

    std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>::iterator itx; 
    for (itx = this -> m_o_graph.begin(); itx != this -> m_o_graph.end(); ++itx){
        std::string key = "Truth Graph Feature: " + itx -> first;
        torch::Tensor* ten = this -> assign_features(itx -> first, graph_enum::truth_graph, data); 
        this -> m_loss -> build_loss_function(std::get<1>(itx -> second)); 
        if (!ten){this -> warning(key + " not found in graph.");}
        else {this -> success(key + " found in graph");}
    }

    for (itx = this -> m_o_node.begin(); itx != this -> m_o_node.end(); ++itx){
        std::string key = "Truth Node Feature: " + itx -> first; 
        torch::Tensor* ten = this -> assign_features(itx -> first, graph_enum::truth_node, data); 
        this -> m_loss -> build_loss_function(std::get<1>(itx -> second)); 
        if (!ten){this -> warning(key + " not found in graph.");}
        else {this -> success(key + " found in graph.");}
    }

    for (itx = this -> m_o_edge.begin(); itx != this -> m_o_edge.end(); ++itx){
        std::string key = "Truth Edge Feature: " + itx -> first; 
        torch::Tensor* ten = this -> assign_features(itx -> first, graph_enum::truth_edge, data);  
        this -> m_loss -> build_loss_function(std::get<1>(itx -> second)); 
        if (!ten){this -> warning(key + " not found in graph");}
        else {this -> success(key + " found in graph");}
    }

    this -> forward(data); 
    this -> train_sequence(true); 
}



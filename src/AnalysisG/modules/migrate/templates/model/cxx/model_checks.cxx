#include <templates/model_template.h>

void model_template::check_features(graph_t* data){
    auto lambda = [this](
            graph_t* data_, graph_enum selx, std::string _name
            std::map<std::string, torch::Tensor*>* fts, 
    ) -> void {
        _name = "Input " + _name + "Feature: "; 

        std::map<std::string, torch::Tensor*>::iterator itr; 
        for (itr = fts -> begin(); itr != fts -> end(); ++itr){
           std::string key = _name + itr -> first; 
           torch::Tensor* ten = this -> assign_features(itr -> first, selx, data_); 
           if (!ten){this -> warning(key + " not found in graph.");}
           else {this -> success(key + " found in graph");}
        }
    }; 

    this -> flush_outputs(); 
    data -> transfer_to_device(this -> m_option); 
    lambda(data, graph_enum::data_graph, "Graph", &this -> m_i_graph); 
    lambda(data, graph_enum::data_node , "Node" , &this -> m_i_node ); 
    lambda(data, graph_enum::data_node , "Edge" , &this -> m_i_edge ); 




    for (itr = this -> m_i_edge.begin(); itr != this -> m_i_edge.end(); ++itr){
        std::string key = "Input Edge Feature: " + itr -> first; 
        torch::Tensor* ten = this -> assign_features(itr -> first, graph_enum::data_edge, data); 
        if (!ten){this -> warning(key + " not found in graph");}
        else {this -> success(key + " found in graph");}
    }

    std::map<std::string, std::tuple<torch::Tensor*, lossfx*>>::iterator itx; 
    for (itx = this -> m_o_graph.begin(); itx != this -> m_o_graph.end(); ++itx){
        std::string key = "Truth Graph Feature: " + itx -> first;
        torch::Tensor* ten = this -> assign_features(itx -> first, graph_enum::truth_graph, data); 
        std::get<1>(itx -> second) -> build_loss_function(); 
        if (!ten){this -> warning(key + " not found in graph.");}
        else {this -> success(key + " found in graph");}
    }

    for (itx = this -> m_o_node.begin(); itx != this -> m_o_node.end(); ++itx){
        std::string key = "Truth Node Feature: " + itx -> first; 
        torch::Tensor* ten = this -> assign_features(itx -> first, graph_enum::truth_node, data); 
        std::get<1>(itx -> second) -> build_loss_function(); 
        if (!ten){this -> warning(key + " not found in graph.");}
        else {this -> success(key + " found in graph.");}
    }

    for (itx = this -> m_o_edge.begin(); itx != this -> m_o_edge.end(); ++itx){
        std::string key = "Truth Edge Feature: " + itx -> first; 
        torch::Tensor* ten = this -> assign_features(itx -> first, graph_enum::truth_edge, data);  
        std::get<1>(itx -> second) -> build_loss_function(); 
        if (!ten){this -> warning(key + " not found in graph");}
        else {this -> success(key + " found in graph");}
    }

    this -> forward(data); 
    this -> train_sequence(true); 
}



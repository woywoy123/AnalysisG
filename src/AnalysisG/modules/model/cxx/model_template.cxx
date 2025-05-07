/**
 * @file model_template.cxx
 * @brief Implementation of the model_template class methods.
 *
 * This file contains the implementation of methods declared in model_template.h,
 * providing functionality for model initialization, module management, feature assignment,
 * and forward propagation in the AnalysisG framework's machine learning component.
 */

#include <templates/model_template.h>

/**
 * @brief Constructor for the `model_template` class.
 * 
 * Sets up property bindings for input and output features, device settings, and initializes
 * the loss function manager. Each property is bound to its appropriate setter and getter methods.
 */
model_template::model_template(){
    // Configure input feature properties
    this -> i_graph.set_setter(this -> set_input_features);
    this -> i_graph.set_object(&this -> m_i_graph); 

    this -> i_node.set_setter(this -> set_input_features);
    this -> i_node.set_object(&this -> m_i_node); 

    this -> i_edge.set_setter(this -> set_input_features);
    this -> i_edge.set_object(&this -> m_i_edge);
   
    // Configure output feature properties
    this -> o_graph.set_setter(this -> set_output_features); 
    this -> o_graph.set_object(&this -> m_o_graph); 

    this -> o_node.set_setter(this -> set_output_features); 
    this -> o_node.set_object(&this -> m_o_node); 

    this -> o_edge.set_setter(this -> set_output_features); 
    this -> o_edge.set_object(&this -> m_o_edge); 

    // Configure device property
    this -> device.set_setter(this -> set_device); 
    this -> device.set_object(this); 
    
    // Configure name property
    this -> name.set_setter(this -> set_name);
    this -> name.set_getter(this -> get_name);
    this -> name.set_object(this); 

    // Configure device index property
    this -> device_index.set_setter(this -> set_dev_index);
    this -> device_index.set_getter(this -> get_dev_index);
    this -> device_index.set_object(this); 

    // Initialize loss function manager
    this -> m_loss = new lossfx(); 
}

/**
 * @brief Creates and returns a clone of this model template.
 * @return A new model_template instance.
 */
model_template* model_template::clone(){return new model_template();}

/**
 * @brief Registers a PyTorch sequential module with the model.
 * @param data Pointer to the sequential module to register.
 * 
 * If a device option is set, the module is transferred to that device before registration.
 */
void model_template::register_module(torch::nn::Sequential* data){
    if (this -> m_option){(*data) -> to(this -> m_option -> device());}
    this -> m_data.push_back(data);
}

/**
 * @brief Registers a PyTorch sequential module with the model and initializes its weights.
 * @param data Pointer to the sequential module to register.
 * @param method The initialization method to use for the module's weights.
 * 
 * If a device option is set, the module is transferred to that device before registration.
 * The module's weights are initialized according to the specified method.
 */
void model_template::register_module(torch::nn::Sequential* data, mlp_init method){
    if (this -> m_option){(*data) -> to(this -> m_option -> device());}
    this -> m_loss -> weight_init(data, method); 
    this -> m_data.push_back(data);
}

/**
 * @brief Default implementation of the forward pass for a single graph.
 * @param data Pointer to the graph_t structure containing input data.
 * 
 * This is a placeholder that derived classes should override with their specific implementation.
 */
void model_template::forward(graph_t*){}

/**
 * @brief Destructor for the model_template class.
 * 
 * Cleans up all resources used by the model, including registered modules,
 * output tensors, and the loss function manager.
 */
model_template::~model_template(){
    for (size_t x(0); x < this -> m_data.size(); ++x){delete this -> m_data[x];}
    this -> flush_outputs(); 
    this -> m_data.clear(); 
    delete this -> m_loss; 
}

/**
 * @brief Assigns features from a graph to tensors based on feature name and type.
 * @param inpt Feature name to assign.
 * @param type Type of graph component (node, edge, or graph).
 * @param data Pointer to the graph_t structure containing the features.
 * @return Pointer to the resulting tensor containing the assigned features.
 * 
 * Extracts specified features from the graph and converts them to PyTorch tensors.
 */
torch::Tensor* model_template::assign_features(std::string inpt, graph_enum type, graph_t* data){
    torch::Tensor* tn = nullptr; 
    switch (type){
        case graph_enum::data_graph:  tn = data -> get_data_graph( inpt, this); this -> m_i_graph[inpt] = tn; break;  
        case graph_enum::data_node:   tn = data -> get_data_node(  inpt, this); this -> m_i_node[inpt]  = tn; break;  
        case graph_enum::data_edge:   tn = data -> get_data_edge(  inpt, this); this -> m_i_edge[inpt]  = tn; break;  
        case graph_enum::truth_graph: tn = data -> get_truth_graph(inpt, this); std::get<0>(this -> m_o_graph[inpt]) = tn; break;  
        case graph_enum::truth_node:  tn = data -> get_truth_node( inpt, this); std::get<0>(this -> m_o_node[inpt])  = tn; break;  
        case graph_enum::truth_edge:  tn = data -> get_truth_edge( inpt, this); std::get<0>(this -> m_o_edge[inpt])  = tn; break;  
        default: return tn; 
    }
    return tn; 
}

/**
 * @brief Assigns features from multiple graphs to tensors based on feature name and type.
 * @param inpt Feature name to assign.
 * @param type Type of graph component (node, edge, or graph).
 * @param data Pointer to a vector of graph_t structures containing the features.
 * @return Pointer to the resulting tensor containing the assigned features.
 * 
 * Extracts specified features from multiple graphs and converts them to batched PyTorch tensors.
 */
torch::Tensor* model_template::assign_features(std::string inpt, graph_enum type, std::vector<graph_t*>* data){
    auto g_data  = [this](graph_t* d, std::string key) -> torch::Tensor* {return d -> get_data_graph(key, this);}; 
    auto n_data  = [this](graph_t* d, std::string key) -> torch::Tensor* {return d -> get_data_node(key, this);}; 
    auto e_data  = [this](graph_t* d, std::string key) -> torch::Tensor* {return d -> get_data_edge(key, this);}; 
    auto g_truth = [this](graph_t* d, std::string key) -> torch::Tensor* {return d -> get_truth_graph(key, this);}; 
    auto n_truth = [this](graph_t* d, std::string key) -> torch::Tensor* {return d -> get_truth_node(key, this);}; 
    auto e_truth = [this](graph_t* d, std::string key) -> torch::Tensor* {return d -> get_truth_edge(key, this);}; 
    auto lamb_d  = [this](
            std::function<torch::Tensor*(graph_t*, std::string)> fx, 
            std::vector<graph_t*>* merge, std::string key) -> torch::Tensor* 
    {
        std::vector<torch::Tensor> arr; 
        for (size_t x(0); x < merge -> size(); ++x){
            torch::Tensor* val = fx((*merge)[x], key);
            if (!val){continue;} 
            arr.push_back(*val); 
        } 
        if (!arr.size()){return nullptr;}
        return new torch::Tensor(torch::cat(arr, {0}));
    }; 

    torch::Tensor* tn = nullptr; 
    switch (type){
        case graph_enum::data_graph:  tn = lamb_d(g_data,  data, inpt); this -> m_i_graph[inpt] = tn; return tn;  
        case graph_enum::data_node:   tn = lamb_d(n_data,  data, inpt); this -> m_i_node[inpt]  = tn; return tn;  
        case graph_enum::data_edge:   tn = lamb_d(e_data,  data, inpt); this -> m_i_edge[inpt]  = tn; return tn;  
        case graph_enum::truth_graph: tn = lamb_d(g_truth, data, inpt); std::get<0>(this -> m_o_graph[inpt]) = tn; return tn;  
        case graph_enum::truth_node:  tn = lamb_d(n_truth, data, inpt); std::get<0>(this -> m_o_node[inpt])  = tn; return tn;  
        case graph_enum::truth_edge:  tn = lamb_d(e_truth, data, inpt); std::get<0>(this -> m_o_edge[inpt])  = tn; return tn;  
        default: return tn; 
    }
    return tn; 
}

/**
 * @brief Performs forward pass on a single graph, with optional training mode.
 * @param data Pointer to the graph_t structure containing input data.
 * @param train Flag indicating if the model is in training mode (true) or evaluation mode (false).
 * 
 * Sets the model's modules to training or evaluation mode as specified, then performs the forward pass.
 */
void model_template::forward(graph_t* data, bool train){
    this -> flush_outputs(); 
    this -> m_batched = false; 

    this -> edge_index = data -> get_edge_index(this); 
    if (this -> m_i_graph.size()){this -> assign(&this -> m_i_graph, graph_enum::data_graph, data);} 
    if (this -> m_i_node.size()){ this -> assign(&this -> m_i_node,  graph_enum::data_node, data);} 
    if (this -> m_i_edge.size()){ this -> assign(&this -> m_i_edge,  graph_enum::data_edge, data);} 

    if (this -> m_o_graph.size()){this -> assign(&this -> m_o_graph, graph_enum::truth_graph, data);}  
    if (this -> m_o_node.size()){ this -> assign(&this -> m_o_node,  graph_enum::truth_node, data);}
    if (this -> m_o_edge.size()){ this -> assign(&this -> m_o_edge,  graph_enum::truth_edge, data);}

    this -> forward(data); 
    this -> train_sequence(train);
}

/**
 * @brief Performs forward pass on multiple graphs, with optional training mode.
 * @param data Vector of pointers to graph_t structures containing input data.
 * @param train Flag indicating if the model is in training mode (true) or evaluation mode (false).
 * 
 * Sets the model's modules to training or evaluation mode as specified, then performs the forward pass
 * on each graph in the input vector.
 */
void model_template::forward(std::vector<graph_t*> data, bool train){
    this -> flush_outputs(); 
    this -> m_batched = true; 

    if (this -> m_i_graph.size()){this -> assign(&this -> m_i_graph, graph_enum::data_graph, &data);} 
    if (this -> m_i_node.size()){ this -> assign(&this -> m_i_node,  graph_enum::data_node, &data);} 
    if (this -> m_i_edge.size()){ this -> assign(&this -> m_i_edge,  graph_enum::data_edge, &data);} 

    if (this -> m_o_graph.size()){this -> assign(&this -> m_o_graph, graph_enum::truth_graph, &data);}  
    if (this -> m_o_node.size()){ this -> assign(&this -> m_o_node,  graph_enum::truth_node, &data);}
    if (this -> m_o_edge.size()){ this -> assign(&this -> m_o_edge,  graph_enum::truth_edge, &data);}

    int offset_nodes = 0; 
    std::vector<torch::Tensor> _edge_index;
    for (size_t x(0); x < data.size(); ++x){
        graph_t* gr = data[x]; 
        if (!_edge_index.size()){_edge_index.push_back(*gr -> get_edge_index(this));}
        else {_edge_index.push_back((*gr -> get_edge_index(this)) + offset_nodes);}
        this -> forward(gr); 
        offset_nodes += gr -> num_nodes; 
    }
    this -> edge_index = new torch::Tensor(torch::cat(_edge_index, {-1})); 
    this -> train_sequence(train);
}

/**
 * @brief Sets input features for a specific feature map.
 * @param inpt Pointer to a vector of strings containing feature names.
 * @param in_fx Pointer to the feature map to set.
 * 
 * Initializes map entries for each feature name in the input vector.
 */
void model_template::set_input_features(std::vector<std::string>* inpt, std::map<std::string, torch::Tensor*>* in_fx){
    for (size_t x(0); x < inpt -> size(); ++x){(*in_fx)[inpt -> at(x)] = nullptr;}
}

/**
 * @brief Clears all output tensors, freeing memory.
 * 
 * Iterates through all output feature maps and deletes their tensor contents.
 */
void model_template::flush_outputs(){
    auto lambda = [](std::map<std::string, torch::Tensor*>* data){
        std::map<std::string, torch::Tensor*>::iterator itr = data -> begin(); 
        for (; itr != data -> end(); ++itr){
            if (!itr -> second){continue;}
            delete itr -> second; 
            itr -> second = nullptr; 
        }
    }; 

    auto lamb = [](std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>* inpx){
        std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>::iterator itx = inpx -> begin(); 
        for (; itx != inpx -> end(); ++itx){
            delete std::get<0>(itx -> second); 
            std::get<0>(itx -> second) = nullptr;
        }
    };


    lambda(&this -> m_p_graph); 
    lambda(&this -> m_p_node); 
    lambda(&this -> m_p_edge); 
    lambda(&this -> m_p_undef); 

    this -> m_p_loss.clear(); 
    if (!this -> m_batched){return;}

    delete this -> edge_index; 
    this -> edge_index = nullptr; 
    lambda(&this -> m_i_graph); 
    lambda(&this -> m_i_node); 
    lambda(&this -> m_i_edge); 

    lamb(&this -> m_o_graph); 
    lamb(&this -> m_o_node);
    lamb(&this -> m_o_edge); 
}



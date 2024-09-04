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

torch::Tensor* model_template::assign_features(std::string inpt, graph_enum type, std::vector<graph_t*> data){
    auto g_data  = [this](torch::Tensor** o, graph_t* d, std::string key){(*o) = d -> get_data_graph(key, this);}; 
    auto n_data  = [this](torch::Tensor** o, graph_t* d, std::string key){(*o) = d -> get_data_node(key, this);}; 
    auto e_data  = [this](torch::Tensor** o, graph_t* d, std::string key){(*o) = d -> get_data_edge(key, this);}; 
    auto g_truth = [this](torch::Tensor** o, graph_t* d, std::string key){(*o) = d -> get_truth_graph(key, this);}; 
    auto n_truth = [this](torch::Tensor** o, graph_t* d, std::string key){(*o) = d -> get_truth_node(key, this);}; 
    auto e_truth = [this](torch::Tensor** o, graph_t* d, std::string key){(*o) = d -> get_truth_edge(key, this);}; 

    auto lamb_d = [this](
            std::function<void(torch::Tensor**, graph_t*, std::string)> fx,
            std::vector<graph_t*>* merge, torch::Tensor** out, std::string key
    ){
        std::vector<torch::Tensor> arr; 
        for (size_t x(0); x < merge -> size(); ++x){
            torch::Tensor* val = nullptr; 
            fx(&val, (*merge)[x], key);
            if (!val){continue;} 
            arr.push_back(*val); 
        } 
        if (!arr.size()){return;}
        (*out) = new torch::Tensor(torch::cat(arr, {0}));
    }; 

    torch::Tensor* tn = nullptr; 
    switch (type){
        case graph_enum::data_graph:  lamb_d(g_data, &data, &tn, inpt); this -> m_i_graph[inpt] = tn; break;  
        case graph_enum::data_node:   lamb_d(n_data, &data, &tn, inpt); this -> m_i_node[inpt]  = tn; break;  
        case graph_enum::data_edge:   lamb_d(e_data, &data, &tn, inpt); this -> m_i_edge[inpt]  = tn; break;  
        case graph_enum::truth_graph: lamb_d(g_truth, &data, &tn, inpt); std::get<0>(this -> m_o_graph[inpt]) = tn; break;  
        case graph_enum::truth_node:  lamb_d(n_truth, &data, &tn, inpt); std::get<0>(this -> m_o_node[inpt])  = tn; break;  
        case graph_enum::truth_edge:  lamb_d(e_truth, &data, &tn, inpt); std::get<0>(this -> m_o_edge[inpt])  = tn; break;  
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
    this -> m_batched = false; 

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

void model_template::forward(std::vector<graph_t*> data, bool train){
    this -> flush_outputs(); 
    this -> m_batched = true; 

    graph_enum mode; 
    std::map<std::string, torch::Tensor*>::iterator itr;
    std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>::iterator itx; 

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
    if (!this -> m_o_graph.count(key)){return this -> warning("Graph Output Feature: " + key + " not found.");}
    if (!this -> m_p_graph[key]){this -> m_p_graph[key] = new torch::Tensor(pred); return;}
    torch::Tensor* tn = this -> m_p_graph[key]; 
    torch::Tensor* tn_ = new torch::Tensor(torch::cat({*tn, pred}, {0})); 
    delete tn; 
    this -> m_p_graph[key] = tn_;
}

void model_template::prediction_node_feature(std::string key, torch::Tensor pred){
    if (!this -> m_o_node.count(key)){return this -> warning("Node Output Feature: " + key + " not found.");}
    if (!this -> m_p_node[key]){this -> m_p_node[key] = new torch::Tensor(pred); return;}
    torch::Tensor* tn = this -> m_p_node[key]; 
    torch::Tensor* tn_ = new torch::Tensor(torch::cat({*tn, pred}, {0})); 
    delete tn; 
    this -> m_p_node[key] = tn_;
}

void model_template::prediction_edge_feature(std::string key, torch::Tensor pred){
    if (!this -> m_o_edge.count(key)){return this -> warning("Edge Output Feature: " + key + " not found.");}
    if (!this -> m_p_edge[key]){this -> m_p_edge[key] = new torch::Tensor(pred); return;}
    torch::Tensor* tn = this -> m_p_edge[key]; 
    torch::Tensor* tn_ = new torch::Tensor(torch::cat({*tn, pred}, {0})); 
    delete tn; 
    this -> m_p_edge[key] = tn_;
}

void model_template::prediction_extra(std::string key, torch::Tensor pred){
    if (!this -> m_p_undef[key]){return;}
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
    if (!this -> m_batched){return;}

    delete this -> edge_index; 
    this -> edge_index = nullptr; 
    lambda(&this -> m_i_graph); 
    lambda(&this -> m_i_node); 
    lambda(&this -> m_i_edge); 

    graph_enum mode; 
    std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>::iterator itx; 
    mode = graph_enum::truth_graph; 
    itx = this -> m_o_graph.begin(); 
    for (; itx != this -> m_o_graph.end(); ++itx){delete std::get<0>(itx -> second); std::get<0>(itx -> second) = nullptr;}

    mode = graph_enum::truth_node; 
    itx = this -> m_o_node.begin(); 
    for (; itx != this -> m_o_node.end(); ++itx){delete std::get<0>(itx -> second); std::get<0>(itx -> second) = nullptr;}

    mode = graph_enum::truth_edge; 
    itx = this -> m_o_edge.begin(); 
    for (; itx != this -> m_o_edge.end(); ++itx){delete std::get<0>(itx -> second); std::get<0>(itx -> second) = nullptr;}

}



#include <templates/model_template.h>

void model_template::clone_settings(model_settings_t* setd){
    setd -> e_optim = this -> e_optim; 
    setd -> s_optim = this -> s_optim; 

    setd -> model_name   = this -> name; 
    setd -> model_device = this -> device;

    setd -> o_graph = this -> o_graph;
    setd -> o_node  = this -> o_node; 
    setd -> o_edge  = this -> o_edge; 

    setd -> i_graph = this -> i_graph; 
    setd -> i_node  = this -> i_node; 
    setd -> i_edge  = this -> i_edge; 
}

void model_template::import_settings(model_settings_t* setd){
    this -> e_optim = setd -> e_optim; 
    this -> s_optim = setd -> s_optim;        
                                             
    this -> name    = setd -> model_name;     
    this -> device  = setd -> model_device;   
                                             
    this -> o_graph = setd -> o_graph;        
    this -> o_node  = setd -> o_node;         
    this -> o_edge  = setd -> o_edge;         
                                            
    this -> i_graph = setd -> i_graph;        
    this -> i_node  = setd -> i_node;         
    this -> i_edge  = setd -> i_edge;         
}

void model_template::set_device(std::string* dev, model_template* md){
    if (md -> op){delete md -> op; md -> op = nullptr;}    
    
    int device_n = -1; 
    c10::DeviceType device_enum; 
    std::string device = md -> lower(dev); 

    if (md -> has_string(&device, "cuda:")){
        device_enum = c10::kCUDA; 
        device_n = std::stoi(md -> split(device, ":")[1]);  
    }
    else if (md -> has_string(&device, "cuda")){device_enum = c10::kCUDA; device_n = 0;}
    else {device_enum = c10::kCPU;} 

    switch(device_enum){
        case c10::kCPU:  md -> op = new torch::TensorOptions(device_enum); break; 
        case c10::kCUDA: md -> op = new torch::TensorOptions(device_enum, device_n); break; 
        default: md -> op = new torch::TensorOptions(device_enum); break; 
    }
    for (size_t x(0); x < md -> m_data.size(); ++x){(*md -> m_data[x]) -> to(md -> op -> device());}
}

void model_template::set_optimizer(std::string name){
    this -> e_optim = opt_from_string(this -> lower(&name)); 
    if (this -> e_optim != opt_enum::invalid_optimizer){}
    else {return this -> failure("Invalid Optimizer");}
    this -> success("Using " + name + " as Optimizer"); 
    this -> s_optim = name;
}

void model_template::initialize(torch::optim::Optimizer** inpt){
    this -> info("------------- Checking Model Parameters ---------------"); 
    if (!this -> m_data.size()){return this -> failure("No parameters defined!");}
    this -> success("OK > Parameters defined."); 

    bool in_feats = this -> m_i_graph.size(); 
    in_feats += this -> m_i_node.size(); 
    in_feats += this -> m_i_edge.size(); 
    if (!in_feats){return this -> failure("No input features defined!");}
    this -> success("OK > Input features defined."); 

    bool o_feats = this -> m_o_graph.size();  
    o_feats += this -> m_o_node.size();  
    o_feats += this -> m_o_edge.size();  
    if (!o_feats){return this -> failure("No ouput features defined!");}
    this -> success("OK > Output features defined."); 

    if (!this -> s_optim.size()){return this -> failure("Failed to register Optimizer!");}
    this -> success("OK > Optimizer defined."); 

    std::vector<torch::Tensor> params = {}; 
    for (size_t x(0); x < this -> m_data.size(); ++x){
        torch::nn::Sequential sq = *this -> m_data[x];
        std::vector<torch::Tensor> p_ = sq.ptr() -> parameters(); 
        params.insert(params.end(), p_.begin(), p_.end()); 
    }
    switch (this -> e_optim){
        case opt_enum::adam   : this -> m_optim = new torch::optim::Adam(params, torch::optim::AdamOptions(1e-6)); break; 
        case opt_enum::adagrad: this -> m_optim = new torch::optim::Adagrad(params); break; 
        case opt_enum::adamw  : this -> m_optim = new torch::optim::AdamW(params); break; 
        case opt_enum::lbfgs  : this -> m_optim = new torch::optim::LBFGS(params); break; 
        case opt_enum::rmsprop: this -> m_optim = new torch::optim::RMSprop(params); break; 
        //case opt_enum::sgd    : this -> m_optim = new torch::optim::SGD((*sq) -> parameters());
        default: this -> failure("Could not initialize the optimizer");
    }

    this -> success("OK > Using device: " + std::string(this -> device)); 
    *inpt = this -> m_optim; 
}



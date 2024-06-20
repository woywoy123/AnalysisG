#include <templates/model_template.h>

void model_template::clone_settings(model_settings_t* setd){
    setd -> e_optim = this -> e_optim; 
    setd -> s_optim = this -> s_optim;
    setd -> model_checkpoint_path = this -> model_checkpoint_path; 

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
    this -> model_checkpoint_path = setd -> model_checkpoint_path; 
                                             
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
    if (md -> m_option){delete md -> m_option; md -> m_option = nullptr;}    
    
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
        case c10::kCPU:  md -> m_option = new torch::TensorOptions(device_enum); break; 
        case c10::kCUDA: md -> m_option = new torch::TensorOptions(device_enum, device_n); break; 
        default: md -> m_option = new torch::TensorOptions(device_enum); break; 
    }
    for (size_t x(0); x < md -> m_data.size(); ++x){(*md -> m_data[x]) -> to(md -> m_option -> device());}
}

void model_template::set_optimizer(std::string name){
    this -> e_optim = this -> m_loss -> optim_string(this -> lower(&name)); 
    if (this -> e_optim != opt_enum::invalid_optimizer){}
    else {return this -> failure("Invalid Optimizer");}
    this -> success("Using " + name + " as Optimizer"); 
    this -> s_optim = name;
}

void model_template::initialize(optimizer_params_t* op_params){
    this -> prefix = this -> name; 
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

    this -> m_optim = this -> m_loss -> build_optimizer(op_params, &params); 
    this -> success("OK > Using device: " + std::string(this -> device)); 
}

void model_template::save_state(){
    torch::serialize::OutputArchive state_session; 
    for (size_t x(0); x < this -> m_data.size(); ++x){(*this -> m_data.at(x)) -> save(state_session);}
    std::string pth = this -> model_checkpoint_path + "epoch-" + std::to_string(this -> epoch) + "/"; 
    this -> create_path(pth); 
    pth += "kfold-" + std::to_string(this -> kfold); 
    state_session.save_to(pth + "_model.pt"); 

    torch::serialize::OutputArchive state_optim; 
    this -> m_optim -> save(state_optim); 
    state_optim.save_to(pth + "_optimizer.pt"); 
}

void model_template::restore_state(){
    torch::serialize::InputArchive state_session; 
    std::string pth = this -> model_checkpoint_path + "epoch-" + std::to_string(this -> epoch) + "/"; 
    pth += "kfold-" + std::to_string(this -> kfold); 
    if (!this -> is_file(pth + "_model.pt")){return;}

    state_session.load_from(pth + "_model.pt");
    for (size_t x(0); x < this -> m_data.size(); ++x){(*this -> m_data.at(x)) -> load(state_session);}
    torch::serialize::InputArchive state_optim; 
    state_optim.load_from(pth + "_optimizer.pt"); 
    this -> m_optim -> load(state_optim); 
    this -> m_optim -> step();  
}

#include <templates/model_template.h>
#include <fstream>

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
    setd -> is_mc   = this -> is_mc; 

    setd -> inference_mode = this -> inference_mode; 
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
    this -> is_mc   = setd -> is_mc; 

    this -> inference_mode = setd -> inference_mode; 
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
    std::string pth = this -> model_checkpoint_path; 
    pth += "state/epoch-" + std::to_string(this -> epoch) + "/"; 
    this -> create_path(pth); 
    pth += "kfold-" + std::to_string(this -> kfold); 

    if (this -> use_pkl){
        std::vector<std::vector<torch::Tensor>> data; 
        for (size_t x(0); x < this -> m_data.size(); ++x){
            data.push_back((*this -> m_data[x]) -> parameters()); 
        }
        std::vector<char> chars = torch::pickle_save(data); 
        std::ofstream ofs((pth + "_model.zip").c_str(), std::ios::out | std::ios::binary); 
        ofs.write(chars.data(), chars.size()); 
        ofs.close(); 
        return; 
    }

    torch::serialize::OutputArchive state_session;
    for (size_t x(0); x < this -> m_data.size(); ++x){(*this -> m_data[x]) -> save(state_session);}
    state_session.save_to(pth + "_model.pt"); 

    torch::serialize::OutputArchive state_optim; 
    this -> m_optim -> save(state_optim); 
    state_optim.save_to(pth + "_optimizer.pt"); 
}

bool model_template::restore_state(){
    std::string model_pth = "";
    std::string optim_pth = ""; 
    if (this -> ends_with(&this -> model_checkpoint_path, ".pt")){
        model_pth = this -> model_checkpoint_path;
        this -> inference_mode = true; 
    }
    else {
        std::string pth = this -> model_checkpoint_path; 
        pth += "state/epoch-" + std::to_string(this -> epoch) + "/"; 
        pth += "kfold-" + std::to_string(this -> kfold); 
        model_pth = pth + "_model.pt"; 
        optim_pth = pth + "_optimizer.pt"; 
    }

    if (!this -> is_file(model_pth)){return false;}
    torch::serialize::InputArchive state_session; 
    state_session.load_from(model_pth, this -> m_option -> device());
    for (size_t x(0); x < this -> m_data.size(); ++x){
        (*this -> m_data[x]) -> load(state_session);
        if (this -> inference_mode){(*this -> m_data[x]) -> eval();}
        else {(*this -> m_data[x]) -> train(true);}
    }

    if (this -> inference_mode){return true;}
    this -> shush = false; 
    this -> success("OK > Found Prior training at: " + model_pth); 
    this -> shush = true; 
    torch::serialize::InputArchive state_optim; 
    state_optim.load_from(optim_pth, this -> m_option -> device()); 
    this -> m_optim -> load(state_optim); 
    this -> m_optim -> step();  
    return true; 
}

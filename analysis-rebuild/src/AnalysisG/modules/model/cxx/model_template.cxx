#include <templates/model_template.h>
#include <torch/torch.h>

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
}

model_template::~model_template(){
    if (this -> op){delete this -> op;}
    std::map<std::string, std::tuple<torch::Tensor*, torch::nn::Module*>>::iterator itr; 
    for (itr = this -> m_o_graph.begin(); itr != this -> m_o_graph.end(); ++itr){
        torch::nn::Module* lss = std::get<1>(itr -> second); 
        if (!lss){continue;}
        delete lss; 
    }

    for (itr = this -> m_o_node.begin(); itr != this -> m_o_node.end(); ++itr){
        torch::nn::Module* lss = std::get<1>(itr -> second); 
        if (!lss){continue;}
        delete lss; 
    }

    for (itr = this -> m_o_edge.begin(); itr != this -> m_o_edge.end(); ++itr){
        torch::nn::Module* lss = std::get<1>(itr -> second); 
        if (!lss){continue;}
        delete lss; 
    }

    if (this -> m_optim){delete this -> m_optim;}
    for (size_t x(0); x < this -> m_data.size(); ++x){delete this -> m_data[x];}
    this -> m_data.clear(); 
}

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

model_template* model_template::clone(){return new model_template();}

void model_template::set_device(std::string* dev, model_template* md){
    if (md -> op){return;}
}

void model_template::register_module(torch::nn::Sequential* data){
    this -> m_data.push_back(data);
}

void model_template::set_optimizer(std::string name){
    if (opt_from_string(this -> lower(&name)) != opt_enum::invalid_optimizer){}
    else {return this -> failure("Invalid Optimizer");}
    this -> success("Using " + name + " as Optimizer"); 
    this -> s_optim = name;
}

void model_template::initialize(torch::optim::Optimizer** inpt){
    this -> info("------------- Checking Model Parameters ---------------"); 
    if (!this -> m_data.size()){return this -> failure("No parameters defined!");}

    bool in_feats = this -> m_i_graph.size(); 
    in_feats += this -> m_i_node.size(); 
    in_feats += this -> m_i_edge.size(); 
    if (!in_feats){return this -> failure("No input features defined!");}

    bool o_feats = this -> m_o_graph.size();  
    o_feats += this -> m_o_node.size();  
    o_feats += this -> m_o_edge.size();  
    if (!o_feats){return this -> failure("No ouput features defined!");}
    if (!this -> s_optim.size()){return this -> failure("Failed to register Optimizer!");}

    for (size_t x(0); x < this -> m_data.size(); ++x){
        torch::nn::Sequential* sq = this -> m_data[x]; 
        if (this -> m_optim){this -> m_optim -> add_parameters((*sq) -> parameters());}
        if (this -> m_optim){continue;}

        switch (this -> e_optim){
            case opt_enum::adam   : this -> m_optim = new torch::optim::Adam((*sq) -> parameters());
            case opt_enum::adagrad: this -> m_optim = new torch::optim::Adagrad((*sq) -> parameters());
            case opt_enum::adamw  : this -> m_optim = new torch::optim::AdamW((*sq) -> parameters());
            case opt_enum::lbfgs  : this -> m_optim = new torch::optim::LBFGS((*sq) -> parameters());
            case opt_enum::rmsprop: this -> m_optim = new torch::optim::RMSprop((*sq) -> parameters());
            //case opt_enum::sgd    : this -> m_optim = new torch::optim::SGD((*sq) -> parameters());
            break;
        }
    }
    *inpt = this -> m_optim; 
}

bool model_template::check_features(graph_t* data){
    std::vector<std::string> gr_i;

    gr_i = this -> i_graph;
    for (std::string n : gr_i){std::cout << data -> get_data_graph(n) << std::endl;}

    gr_i = this -> i_node; 
    for (std::string n : gr_i){
        std::cout << n << std::endl;
        std::cout << data << std::endl; 
        std::cout << data -> get_data_node(n) << std::endl; 
    }





    abort(); 
}

void model_template::forward(graph_t* data){}

void model_template::set_input_features(std::vector<std::string>* inpt, std::map<std::string, torch::Tensor*>* in_fx){
    for (size_t x(0); x < inpt -> size(); ++x){
        (*in_fx)["D-" + inpt -> at(x)] = nullptr;
    }
}


void model_template::set_output_features(
        std::map<std::string, std::string>* inpt, 
        std::map<std::string, std::tuple<torch::Tensor*, torch::nn::Module*>>* out_fx
){
    notification nx = notification(); 
    std::map<std::string, std::string>::iterator itx = inpt -> begin();
    for (; itx != inpt -> end(); ++itx){
        std::string o_fx = itx -> first; 
        std::string l_fx = itx -> second;

        torch::nn::Module* lss = nullptr; 
        switch (loss_from_string(tools().lower(&l_fx))){
            case loss_enum::bce                         : lss = new torch::nn::BCELossImpl(); break;
            case loss_enum::bce_with_logits             : lss = new torch::nn::BCEWithLogitsLossImpl(); break;
            case loss_enum::cosine_embedding            : lss = new torch::nn::CosineEmbeddingLossImpl(); break;
            case loss_enum::cross_entropy               : lss = new torch::nn::CrossEntropyLossImpl(); break;
            case loss_enum::ctc                         : lss = new torch::nn::CTCLossImpl(); break;
            case loss_enum::hinge_embedding             : lss = new torch::nn::HingeEmbeddingLossImpl(); break;
            case loss_enum::huber                       : lss = new torch::nn::HuberLossImpl(); break;
            case loss_enum::kl_div                      : lss = new torch::nn::KLDivLossImpl(); break;
            case loss_enum::l1                          : lss = new torch::nn::L1LossImpl(); break;
            case loss_enum::margin_ranking              : lss = new torch::nn::MarginRankingLossImpl(); break;
            case loss_enum::mse                         : lss = new torch::nn::MSELossImpl(); break;
            case loss_enum::multi_label_margin          : lss = new torch::nn::MultiLabelMarginLossImpl(); break;
            case loss_enum::multi_label_soft_margin     : lss = new torch::nn::MultiLabelSoftMarginLossImpl(); break;
            case loss_enum::multi_margin                : lss = new torch::nn::MultiMarginLossImpl(); break;
            case loss_enum::nll                         : lss = new torch::nn::NLLLossImpl(); break;
            case loss_enum::poisson_nll                 : lss = new torch::nn::PoissonNLLLossImpl(); break;
            case loss_enum::smooth_l1                   : lss = new torch::nn::SmoothL1LossImpl(); break;
            case loss_enum::soft_margin                 : lss = new torch::nn::SoftMarginLossImpl(); break;
            case loss_enum::triplet_margin              : lss = new torch::nn::TripletMarginLossImpl(); break;
            case loss_enum::triplet_margin_with_distance: lss = new torch::nn::TripletMarginWithDistanceLossImpl(); break;
            default: nx.warning("Invalid Loss Function for: " + o_fx + " feature!"); break; 
        }
        (*out_fx)["T-" + o_fx] = {nullptr, lss}; 
        nx.success("Added loss function: " + l_fx + " for " + o_fx);
    }
}












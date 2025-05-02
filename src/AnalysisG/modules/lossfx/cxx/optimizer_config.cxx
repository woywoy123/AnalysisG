#include <templates/lossfx.h>

void lossfx::build_adam(optimizer_params_t* op, std::vector<torch::Tensor>* params){
    if (this -> m_adam){return;}
    torch::optim::AdamOptions optim_(op -> lr); 
    if (op -> m_betas)       {optim_.betas(std::tuple<float, float>(op -> betas));}
    if (op -> m_eps)         {optim_.eps(op -> eps);}
    if (op -> m_weight_decay){optim_.weight_decay(op -> weight_decay);}
    if (op -> m_amsgrad)     {optim_.amsgrad(op -> amsgrad);}
    this -> m_adam = new torch::optim::Adam(*params, optim_); 

    this -> success("Adam Parameters:"); 
    this -> success("Learning Rate: " + std::to_string(op -> lr));
    if (op -> m_betas){
        this -> success("Betas: " + std::to_string(std::get<0>(std::tuple<float, float>(op -> betas))) + "-" +
                                    std::to_string(std::get<1>(std::tuple<float, float>(op -> betas))));
    }
    if (op -> m_eps)         {this -> success("Eps: "          + std::to_string(op -> eps));}
    if (op -> m_weight_decay){this -> success("Weight Decay: " + std::to_string(op -> weight_decay));}
    if (op -> m_amsgrad)     {this -> success("AMS Grad: "     + std::to_string(op -> amsgrad));}
    this -> build_scheduler(op, this -> m_adam); 
}

void lossfx::build_adagrad(optimizer_params_t* op, std::vector<torch::Tensor>* params){
    if (this -> m_adagrad){return;}
    torch::optim::AdagradOptions optim_(op -> lr); 
    if (op -> m_lr_decay)                 {optim_.lr_decay(op -> lr_decay);}
    if (op -> m_weight_decay)             {optim_.weight_decay(op -> weight_decay);}
    if (op -> m_initial_accumulator_value){optim_.initial_accumulator_value(op -> initial_accumulator_value);}
    if (op -> m_eps)                      {optim_.eps(op -> eps);}

    this -> success("ADA Grad Parameters:"); 
    this -> success("Learning Rate: " + std::to_string(op -> lr));
    if (op -> m_lr_decay)                 {this -> success("Learning Rate Decay: " + std::to_string(op -> lr_decay));}
    if (op -> m_weight_decay)             {this -> success("Weight Decay: " + std::to_string(op -> weight_decay));}
    if (op -> m_initial_accumulator_value){this -> success("IAV: " + std::to_string(op -> initial_accumulator_value));}
    if (op -> m_eps)                      {this -> success("Eps: " + std::to_string(op -> eps));}
    this -> m_adagrad = new torch::optim::Adagrad(*params, optim_); 
    this -> build_scheduler(op, this -> m_adagrad); 
}

void lossfx::build_adamw(optimizer_params_t* op, std::vector<torch::Tensor>* params){
    if (this -> m_adamw){return;}
    torch::optim::AdamWOptions optim_(op -> lr); 
    if (op -> m_betas)       {optim_.betas(std::tuple<float, float>(op -> betas));}
    if (op -> m_eps)         {optim_.eps(op -> eps);}
    if (op -> m_weight_decay){optim_.weight_decay(op -> weight_decay);}
    if (op -> m_amsgrad)     {optim_.amsgrad(op -> amsgrad);}

    this -> success("AdamW Parameters:"); 
    this -> success("Learning Rate: " + std::to_string(op -> lr));
    if (op -> m_betas){
        this -> success("Betas: " + std::to_string(std::get<0>(std::tuple<float, float>(op -> betas))) + "-" +
                                    std::to_string(std::get<1>(std::tuple<float, float>(op -> betas))));
    }
    if (op -> m_eps)         {this -> success("Eps: "          + std::to_string(op -> eps));}
    if (op -> m_weight_decay){this -> success("Weight Decay: " + std::to_string(op -> weight_decay));}
    if (op -> m_amsgrad)     {this -> success("AMS Grad: "     + std::to_string(op -> amsgrad));}
    this -> m_adamw = new torch::optim::AdamW(*params, optim_); 
    this -> build_scheduler(op, this -> m_adamw); 
}

void lossfx::build_lbfgs(optimizer_params_t* op, std::vector<torch::Tensor>* params){
    if (this -> m_lbfgs){return;}
    torch::optim::LBFGSOptions optim_(op -> lr); 
    if (op -> m_max_iter)        {optim_.max_iter(op -> max_iter);}
    if (op -> m_tolerance_grad)  {optim_.tolerance_grad(op -> tolerance_grad);} 
    if (op -> m_tolerance_change){optim_.tolerance_change(op -> tolerance_change);}
    if (op -> m_max_iter)        {optim_.max_iter(op -> max_iter);}
    if (op -> m_max_eval)        {optim_.max_eval(op -> max_eval);}
    if (op -> m_history_size)    {optim_.history_size(op -> history_size);}
    this -> m_lbfgs = new torch::optim::LBFGS(*params, optim_); 
    this -> build_scheduler(op, this -> m_lbfgs); 
}

void lossfx::build_rmsprop(optimizer_params_t* op, std::vector<torch::Tensor>* params){
    if (this -> m_rmsprop){return;}
    torch::optim::RMSpropOptions optim_(op -> lr); 
    if (op -> m_alpha)       {optim_.alpha(op -> alpha);}
    if (op -> m_eps)         {optim_.eps(op -> eps);}
    if (op -> m_weight_decay){optim_.weight_decay(op -> weight_decay);}
    if (op -> m_momentum)    {optim_.momentum(op -> momentum);}
    if (op -> m_centered)    {optim_.centered(op -> centered);}
 
    this -> m_rmsprop = new torch::optim::RMSprop(*params, optim_); 
    this -> build_scheduler(op, this -> m_rmsprop); 
}

void lossfx::build_sgd(optimizer_params_t* op, std::vector<torch::Tensor>* params){
    if (this -> m_sgd){return;}
    torch::optim::SGDOptions optim_(op -> lr); 
    if (op -> m_momentum)    {optim_.momentum(op -> momentum);}
    if (op -> m_dampening)   {optim_.dampening(op -> dampening);}
    if (op -> m_weight_decay){optim_.weight_decay(op -> weight_decay);}
    if (op -> m_nesterov)    {optim_.nesterov(op -> nesterov);}

    this -> success("Stochastic Gradient Descent Parameters:"); 
    this -> success("Learning Rate: " + std::to_string(op -> lr));
    if (op -> m_momentum)    {this -> success("Momentum: "     + std::to_string(op -> momentum));}
    if (op -> m_dampening)   {this -> success("Dampening: "    + std::to_string(op -> dampening));}
    if (op -> m_weight_decay){this -> success("Weight Decay: " + std::to_string(op -> weight_decay));}
    if (op -> m_nesterov)    {this -> success("Nesterov: "     + std::to_string(op -> nesterov));}

    this -> m_sgd = new torch::optim::SGD(*params, optim_); 
    this -> build_scheduler(op, this -> m_sgd); 
}



// ------------------------ weight init ------------------------ //
static void m_xavier_normal(torch::nn::Module& m){
     torch::NoGradGuard nograd; 
     if (auto* linear = m.as<torch::nn::Linear>()){
         torch::nn::init::xavier_normal_(linear -> weight);
     }
}

static void m_xavier_uniform(torch::nn::Module& m){
     torch::NoGradGuard nograd; 
     if (auto* linear = m.as<torch::nn::Linear>()){
         torch::nn::init::xavier_uniform_(linear -> weight);
     }
}

static void m_normal(torch::nn::Module& m){
     torch::NoGradGuard nograd; 
     if (auto* linear = m.as<torch::nn::Linear>()){
         torch::nn::init::normal_(linear -> weight);
     }
}

static void m_uniform(torch::nn::Module& m){
     torch::NoGradGuard nograd; 
     if (auto* linear = m.as<torch::nn::Linear>()){
         torch::nn::init::uniform_(linear -> weight);
     }
}


static void m_kaiming_normal(torch::nn::Module& m){
     torch::NoGradGuard nograd; 
     if (auto* linear = m.as<torch::nn::Linear>()){
         torch::nn::init::kaiming_normal_(linear -> weight);
     }
}

static void m_kaiming_uniform(torch::nn::Module& m){
     torch::NoGradGuard nograd; 
     if (auto* linear = m.as<torch::nn::Linear>()){
         torch::nn::init::kaiming_uniform_(linear -> weight);
     }
}

void lossfx::weight_init(torch::nn::Sequential* data, mlp_init method){
    switch (method){
        case mlp_init::normal:          (*data) -> apply(m_normal); break;  
        case mlp_init::uniform:         (*data) -> apply(m_uniform); break; 
        case mlp_init::xavier_normal:   (*data) -> apply(m_xavier_normal); break;  
        case mlp_init::xavier_uniform:  (*data) -> apply(m_xavier_uniform); break; 
        case mlp_init::kaiming_normal:  (*data) -> apply(m_kaiming_normal); break;  
        case mlp_init::kaiming_uniform: (*data) -> apply(m_kaiming_uniform); break; 
        default: break; 
    }
}










#include <templates/lossfx.h>

lossfx::lossfx(){this -> prefix = "optimizer";}
lossfx::lossfx(std::string var, std::string val){
    this -> prefix = "optimizer";
    this -> variable = var; 
    this -> interpret(&val); 
}

lossfx::~lossfx(){
    if (this -> m_adam                        ){delete this -> m_adam                        ;}
    if (this -> m_adagrad                     ){delete this -> m_adagrad                     ;}
    if (this -> m_adamw                       ){delete this -> m_adamw                       ;} 
    if (this -> m_lbfgs                       ){delete this -> m_lbfgs                       ;}
    if (this -> m_rmsprop                     ){delete this -> m_rmsprop                     ;} 
    if (this -> m_sgd                         ){delete this -> m_sgd                         ;}
    if (this -> m_bce                         ){delete this -> m_bce                         ;}
    if (this -> m_bce_with_logits             ){delete this -> m_bce_with_logits             ;}
    if (this -> m_cosine_embedding            ){delete this -> m_cosine_embedding            ;}
    if (this -> m_cross_entropy               ){delete this -> m_cross_entropy               ;}
    if (this -> m_ctc                         ){delete this -> m_ctc                         ;}
    if (this -> m_hinge_embedding             ){delete this -> m_hinge_embedding             ;}
    if (this -> m_huber                       ){delete this -> m_huber                       ;}
    if (this -> m_kl_div                      ){delete this -> m_kl_div                      ;}
    if (this -> m_l1                          ){delete this -> m_l1                          ;}
    if (this -> m_margin_ranking              ){delete this -> m_margin_ranking              ;}
    if (this -> m_mse                         ){delete this -> m_mse                         ;}
    if (this -> m_multi_label_margin          ){delete this -> m_multi_label_margin          ;}
    if (this -> m_multi_label_soft_margin     ){delete this -> m_multi_label_soft_margin     ;}
    if (this -> m_multi_margin                ){delete this -> m_multi_margin                ;}
    if (this -> m_nll                         ){delete this -> m_nll                         ;}
    if (this -> m_poisson_nll                 ){delete this -> m_poisson_nll                 ;}
    if (this -> m_smooth_l1                   ){delete this -> m_smooth_l1                   ;}
    if (this -> m_soft_margin                 ){delete this -> m_soft_margin                 ;}
    if (this -> m_triplet_margin              ){delete this -> m_triplet_margin              ;}
    if (this -> m_triplet_margin_with_distance){delete this -> m_triplet_margin_with_distance;}

    if (this -> m_steplr){delete this -> m_steplr;}
    if (this -> m_rlp   ){delete this -> m_rlp   ;}
    if (this -> m_lrs   ){delete this -> m_lrs   ;}
}

torch::optim::Optimizer* lossfx::build_optimizer(optimizer_params_t* op, std::vector<torch::Tensor>* params){
    opt_enum op_ = this -> optim_string(op -> optimizer); 

    switch (op_){
        case opt_enum::adam   : this -> build_adam(op, params);    return this -> m_adam;       
        case opt_enum::adagrad: this -> build_adagrad(op, params); return this -> m_adagrad;    
        case opt_enum::adamw  : this -> build_adamw(op, params);   return this -> m_adamw;    
        case opt_enum::lbfgs  : this -> build_lbfgs(op, params);   return this -> m_lbfgs;    
        case opt_enum::rmsprop: this -> build_rmsprop(op, params); return this -> m_rmsprop;  
        case opt_enum::sgd    : this -> build_sgd(op, params);     return this -> m_sgd;      
        default: return nullptr; 
    }
    return nullptr; 
}

void lossfx::build_scheduler(optimizer_params_t* op, torch::optim::Optimizer* opx){
    scheduler_enum spx_ = scheduler_string(op -> scheduler);
    if (op -> scheduler.size() && spx_ == scheduler_enum::invalid_scheduler){this -> failure("Invalid Scheduler: " + op -> scheduler);}
    switch(spx_){
        case scheduler_enum::steplr                    : this -> m_steplr = (this -> m_steplr) ? this -> m_steplr : new torch::optim::StepLR(*opx, op -> step_size, op -> gamma); return; 
        case scheduler_enum::reducelronplateauscheduler: this -> m_rlp    = (this -> m_rlp   ) ? this -> m_rlp    : new torch::optim::ReduceLROnPlateauScheduler(*opx); return; 
        default: return;
    }
}

void lossfx::step(){
    if (this -> m_steplr){this -> m_steplr -> step();}
//    if (this -> m_rlp   ){this -> m_rlp -> step();}
}

bool lossfx::build_loss_function(){return this -> build_loss_function(this -> lss_cfg.fx);}

bool lossfx::build_loss_function(loss_enum lss){
    switch (lss){
        case loss_enum::bce                         : this -> build_fx_loss(this -> m_bce                         ); return true; 
        case loss_enum::bce_with_logits             : this -> build_fx_loss(this -> m_bce_with_logits             ); return true; 
        case loss_enum::cosine_embedding            : this -> build_fx_loss(this -> m_cosine_embedding            ); return true; 
        case loss_enum::cross_entropy               : this -> build_fx_loss(this -> m_cross_entropy               ); return true; 
        case loss_enum::ctc                         : this -> build_fx_loss(this -> m_ctc                         ); return true; 
        case loss_enum::hinge_embedding             : this -> build_fx_loss(this -> m_hinge_embedding             ); return true; 
        case loss_enum::huber                       : this -> build_fx_loss(this -> m_huber                       ); return true; 
        case loss_enum::kl_div                      : this -> build_fx_loss(this -> m_kl_div                      ); return true; 
        case loss_enum::l1                          : this -> build_fx_loss(this -> m_l1                          ); return true; 
        case loss_enum::margin_ranking              : this -> build_fx_loss(this -> m_margin_ranking              ); return true; 
        case loss_enum::mse                         : this -> build_fx_loss(this -> m_mse                         ); return true; 
        case loss_enum::multi_label_margin          : this -> build_fx_loss(this -> m_multi_label_margin          ); return true; 
        case loss_enum::multi_label_soft_margin     : this -> build_fx_loss(this -> m_multi_label_soft_margin     ); return true; 
        case loss_enum::multi_margin                : this -> build_fx_loss(this -> m_multi_margin                ); return true; 
        case loss_enum::nll                         : this -> build_fx_loss(this -> m_nll                         ); return true; 
        case loss_enum::poisson_nll                 : this -> build_fx_loss(this -> m_poisson_nll                 ); return true; 
        case loss_enum::smooth_l1                   : this -> build_fx_loss(this -> m_smooth_l1                   ); return true; 
        case loss_enum::soft_margin                 : this -> build_fx_loss(this -> m_soft_margin                 ); return true; 
        case loss_enum::triplet_margin              : this -> build_fx_loss(this -> m_triplet_margin              ); return true; 
        case loss_enum::triplet_margin_with_distance: this -> build_fx_loss(this -> m_triplet_margin_with_distance); return true; 
        default: break; 
    }
    this -> warning("Invalid Loss Function! \nOptions Are:"); 
    this -> warning(" -> bceloss"                      ); 
    this -> warning(" -> bcewithlogitsloss"            ); 
    this -> warning(" -> cosineembeddingloss"          ); 
    this -> warning(" -> crossentropyloss"             ); 
    this -> warning(" -> ctcloss"                      ); 
    this -> warning(" -> hingeembeddingloss"           ); 
    this -> warning(" -> huberloss"                    ); 
    this -> warning(" -> kldivloss"                    ); 
    this -> warning(" -> l1loss"                       ); 
    this -> warning(" -> marginrankingloss"            ); 
    this -> warning(" -> mseloss"                      ); 
    this -> warning(" -> multilabelmarginloss"         ); 
    this -> warning(" -> multilabelsoftmarginloss"     ); 
    this -> warning(" -> multimarginloss"              ); 
    this -> warning(" -> nllloss"                      ); 
    this -> warning(" -> poissonnllloss"               ); 
    this -> warning(" -> smoothl1loss"                 ); 
    this -> warning(" -> softmarginloss"               ); 
    this -> warning(" -> tripletmarginloss"            ); 
    this -> warning(" -> tripletmarginwithdistanceloss");
    return false;
}

void lossfx::to(torch::TensorOptions* op){
    if (this -> m_bce                         ){this -> m_bce                          -> to(op -> device(), true);}
    if (this -> m_bce_with_logits             ){this -> m_bce_with_logits              -> to(op -> device(), true);}
    if (this -> m_cosine_embedding            ){this -> m_cosine_embedding             -> to(op -> device(), true);}
    if (this -> m_cross_entropy               ){this -> m_cross_entropy                -> to(op -> device(), true);}
    if (this -> m_ctc                         ){this -> m_ctc                          -> to(op -> device(), true);}
    if (this -> m_hinge_embedding             ){this -> m_hinge_embedding              -> to(op -> device(), true);}
    if (this -> m_huber                       ){this -> m_huber                        -> to(op -> device(), true);}
    if (this -> m_kl_div                      ){this -> m_kl_div                       -> to(op -> device(), true);}
    if (this -> m_l1                          ){this -> m_l1                           -> to(op -> device(), true);}
    if (this -> m_margin_ranking              ){this -> m_margin_ranking               -> to(op -> device(), true);}
    if (this -> m_mse                         ){this -> m_mse                          -> to(op -> device(), true);}
    if (this -> m_multi_label_margin          ){this -> m_multi_label_margin           -> to(op -> device(), true);}
    if (this -> m_multi_label_soft_margin     ){this -> m_multi_label_soft_margin      -> to(op -> device(), true);}
    if (this -> m_multi_margin                ){this -> m_multi_margin                 -> to(op -> device(), true);}
    if (this -> m_nll                         ){this -> m_nll                          -> to(op -> device(), true);}
    if (this -> m_poisson_nll                 ){this -> m_poisson_nll                  -> to(op -> device(), true);}
    if (this -> m_smooth_l1                   ){this -> m_smooth_l1                    -> to(op -> device(), true);}
    if (this -> m_soft_margin                 ){this -> m_soft_margin                  -> to(op -> device(), true);}
    if (this -> m_triplet_margin              ){this -> m_triplet_margin               -> to(op -> device(), true);}
    if (this -> m_triplet_margin_with_distance){this -> m_triplet_margin_with_distance -> to(op -> device(), true);}
}












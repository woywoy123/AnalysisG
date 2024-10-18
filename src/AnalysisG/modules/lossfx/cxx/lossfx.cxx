#include <templates/lossfx.h>

lossfx::lossfx(){}
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

bool lossfx::build_loss_function(loss_enum lss){
    switch (lss){
        case loss_enum::bce                    : this -> m_bce                     = (this -> m_bce                    ) ? this -> m_bce                    : new torch::nn::BCELossImpl();                  return true;
        case loss_enum::bce_with_logits        : this -> m_bce_with_logits         = (this -> m_bce_with_logits        ) ? this -> m_bce_with_logits        : new torch::nn::BCEWithLogitsLossImpl();        return true;
        case loss_enum::cosine_embedding       : this -> m_cosine_embedding        = (this -> m_cosine_embedding       ) ? this -> m_cosine_embedding       : new torch::nn::CosineEmbeddingLossImpl();      return true;
        case loss_enum::cross_entropy          : this -> m_cross_entropy           = (this -> m_cross_entropy          ) ? this -> m_cross_entropy          : new torch::nn::CrossEntropyLossImpl();         return true;
        case loss_enum::ctc                    : this -> m_ctc                     = (this -> m_ctc                    ) ? this -> m_ctc                    : new torch::nn::CTCLossImpl();                  return true;
        case loss_enum::hinge_embedding        : this -> m_hinge_embedding         = (this -> m_hinge_embedding        ) ? this -> m_hinge_embedding        : new torch::nn::HingeEmbeddingLossImpl();       return true;
        case loss_enum::huber                  : this -> m_huber                   = (this -> m_huber                  ) ? this -> m_huber                  : new torch::nn::HuberLossImpl();                return true;
        case loss_enum::kl_div                 : this -> m_kl_div                  = (this -> m_kl_div                 ) ? this -> m_kl_div                 : new torch::nn::KLDivLossImpl();                return true;
        case loss_enum::l1                     : this -> m_l1                      = (this -> m_l1                     ) ? this -> m_l1                     : new torch::nn::L1LossImpl();                   return true;
        case loss_enum::margin_ranking         : this -> m_margin_ranking          = (this -> m_margin_ranking         ) ? this -> m_margin_ranking         : new torch::nn::MarginRankingLossImpl();        return true;
        case loss_enum::mse                    : this -> m_mse                     = (this -> m_mse                    ) ? this -> m_mse                    : new torch::nn::MSELossImpl();                  return true;
        case loss_enum::multi_label_margin     : this -> m_multi_label_margin      = (this -> m_multi_label_margin     ) ? this -> m_multi_label_margin     : new torch::nn::MultiLabelMarginLossImpl();     return true;
        case loss_enum::multi_label_soft_margin: this -> m_multi_label_soft_margin = (this -> m_multi_label_soft_margin) ? this -> m_multi_label_soft_margin: new torch::nn::MultiLabelSoftMarginLossImpl(); return true;
        case loss_enum::multi_margin           : this -> m_multi_margin            = (this -> m_multi_margin           ) ? this -> m_multi_margin           : new torch::nn::MultiMarginLossImpl();          return true;
        case loss_enum::nll                    : this -> m_nll                     = (this -> m_nll                    ) ? this -> m_nll                    : new torch::nn::NLLLossImpl();                  return true;
        case loss_enum::poisson_nll            : this -> m_poisson_nll             = (this -> m_poisson_nll            ) ? this -> m_poisson_nll            : new torch::nn::PoissonNLLLossImpl();           return true;
        case loss_enum::smooth_l1              : this -> m_smooth_l1               = (this -> m_smooth_l1              ) ? this -> m_smooth_l1              : new torch::nn::SmoothL1LossImpl();             return true;
        case loss_enum::soft_margin            : this -> m_soft_margin             = (this -> m_soft_margin            ) ? this -> m_soft_margin            : new torch::nn::SoftMarginLossImpl();           return true;
        case loss_enum::triplet_margin         : this -> m_triplet_margin          = (this -> m_triplet_margin         ) ? this -> m_triplet_margin         : new torch::nn::TripletMarginLossImpl();        return true;
        case loss_enum::triplet_margin_with_distance: 
            this -> m_triplet_margin_with_distance = (this -> m_triplet_margin_with_distance) ? this -> m_triplet_margin_with_distance: new torch::nn::TripletMarginWithDistanceLossImpl(); return true;
        default: break; 
    }
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












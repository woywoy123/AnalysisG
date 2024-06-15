#ifndef MODEL_ENUM_FUNCTIONS_H
#define MODEL_ENUM_FUNCTIONS_H
#include <string>

// optimizers
enum opt_enum {
    adam, adagrad, adamw, lbfgs, rmsprop, sgd, invalid_optimizer
}; 

// optimizers - parameters
enum opt_para_enum {
    lr, lr_decay, weight_decay, initial_accumulator_value, eps, betas, amsgrad, 
    momentum, dampening, nesterov, alpha, centered, max_iter, max_eval, tolerance_grad, tolerance_change, history_size
}; 

// loss functions
enum loss_enum {
    bce, 
    bce_with_logits, 
    cosine_embedding, 
    cross_entropy, 
    ctc, 
    hinge_embedding, 
    huber, 
    kl_div, 
    l1, 
    margin_ranking, 
    mse, 
    multi_label_margin, 
    multi_label_soft_margin, 
    multi_margin, 
    nll, 
    poisson_nll, 
    smooth_l1, 
    soft_margin, 
    triplet_margin, 
    triplet_margin_with_distance,
    invalid_loss
};


static void adagrad_board(opt_para_enum op){
    switch (op){
        case opt_para_enum::lr                       : return; 
        case opt_para_enum::lr_decay                 : return;
        case opt_para_enum::weight_decay             : return;
        case opt_para_enum::initial_accumulator_value: return; 
        case opt_para_enum::eps                      : return; 
        default: return; 
    }
}; 

static void adam_board(opt_para_enum op){
    switch (op){
        case opt_para_enum::lr          : return; 
        case opt_para_enum::betas       : return;
        case opt_para_enum::eps         : return;
        case opt_para_enum::weight_decay: return;
        case opt_para_enum::amsgrad     : return; 
        default: return; 
    }
}; 

static void adamw_board(opt_para_enum op){
    switch (op){
        case opt_para_enum::lr          : return; 
        case opt_para_enum::betas       : return;
        case opt_para_enum::eps         : return;
        case opt_para_enum::weight_decay: return;
        case opt_para_enum::amsgrad     : return; 
        default: return; 
    }
}; 

static void sgd_board(opt_para_enum op){
    switch (op){
        case opt_para_enum::lr          : return; 
        case opt_para_enum::momentum    : return; 
        case opt_para_enum::dampening   : return; 
        case opt_para_enum::weight_decay: return;
        case opt_para_enum::nesterov    : return;
        default: return; 
    }
}; 


static void rmsprop_board(opt_para_enum op){
    switch (op){
        case opt_para_enum::lr          : return; 
        case opt_para_enum::alpha       : return; 
        case opt_para_enum::eps         : return; 
        case opt_para_enum::weight_decay: return;
        case opt_para_enum::momentum    : return;
        case opt_para_enum::centered    : return;
        default: return; 
    }
}; 

static void lbfgs_board(opt_para_enum op){
    switch (op){
        case opt_para_enum::lr              : return; 
        case opt_para_enum::max_iter        : return; 
        case opt_para_enum::max_eval        : return; 
        case opt_para_enum::tolerance_grad  : return;
        case opt_para_enum::tolerance_change: return;
        case opt_para_enum::history_size    : return;
        default: return; 
    }
}; 

static void switch_board(opt_enum op, opt_para_enum para){
    switch (op){
        case opt_enum::adam:    return adam_board(para);
        case opt_enum::adagrad: return adagrad_board(para);
        case opt_enum::adamw:   return adamw_board(para);
        case opt_enum::lbfgs:   return lbfgs_board(para);
        case opt_enum::rmsprop: return rmsprop_board(para);
        case opt_enum::sgd:     return sgd_board(para);
        default: break;
    }
}; 

static opt_enum opt_from_string(std::string name){
    if (name == "adam"   ){return opt_enum::adam;}
    if (name == "adagrad"){return opt_enum::adagrad;}
    if (name == "adamw"  ){return opt_enum::adamw;}
    if (name == "lbfgs"  ){return opt_enum::lbfgs;}
    if (name == "rmsprop"){return opt_enum::rmsprop;}
    if (name == "sgd"    ){return opt_enum::sgd;}
    return opt_enum::invalid_optimizer;
}; 

static loss_enum loss_from_string(std::string name){
    if(name == "bceloss"                      ){return loss_enum::bce;}
    if(name == "bcewithlogitsloss"            ){return loss_enum::bce_with_logits;}
    if(name == "cosineembeddingloss"          ){return loss_enum::cosine_embedding;}
    if(name == "crossentropyloss"             ){return loss_enum::cross_entropy;}
    if(name == "ctcloss"                      ){return loss_enum::ctc;}
    if(name == "hingeembeddingloss"           ){return loss_enum::hinge_embedding;}
    if(name == "huberloss"                    ){return loss_enum::huber;}
    if(name == "kldivloss"                    ){return loss_enum::kl_div;}
    if(name == "l1loss"                       ){return loss_enum::l1;}
    if(name == "marginrankingloss"            ){return loss_enum::margin_ranking;}
    if(name == "mseloss"                      ){return loss_enum::mse;}
    if(name == "multilabelmarginloss"         ){return loss_enum::multi_label_margin;}
    if(name == "multilabelsoftmarginloss"     ){return loss_enum::multi_label_soft_margin;}
    if(name == "multimarginloss"              ){return loss_enum::multi_margin;}
    if(name == "nllloss"                      ){return loss_enum::nll;}
    if(name == "poissonnllloss"               ){return loss_enum::poisson_nll;}
    if(name == "smoothl1loss"                 ){return loss_enum::smooth_l1;}
    if(name == "softmarginloss"               ){return loss_enum::soft_margin;}
    if(name == "tripletmarginloss"            ){return loss_enum::triplet_margin;}
    if(name == "tripletmarginwithdistanceloss"){return loss_enum::triplet_margin_with_distance;}
    return loss_enum::invalid_loss; 
}; 

#endif

#include <templates/lossfx.h>

loss_enum lossfx::loss_string(std::string name){
    name = this -> lower(&name); 
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
}

opt_enum lossfx::optim_string(std::string name){
    name = this -> lower(&name);
    if (name == "adam"   ){return opt_enum::adam;}
    if (name == "adagrad"){return opt_enum::adagrad;}
    if (name == "adamw"  ){return opt_enum::adamw;}
    if (name == "lbfgs"  ){return opt_enum::lbfgs;}
    if (name == "rmsprop"){return opt_enum::rmsprop;}
    if (name == "sgd"    ){return opt_enum::sgd;}
    return opt_enum::invalid_optimizer;
}


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

scheduler_enum lossfx::scheduler_string(std::string name){
    name = this -> lower(&name);
    if (name == "steplr" ){return scheduler_enum::steplr;}
    if (name == "reducelronplateauscheduler"){return scheduler_enum::reducelronplateauscheduler;}
    if (name == "lrscheduler"  ){return scheduler_enum::lrscheduler;}
    return scheduler_enum::invalid_scheduler;
}

void lossfx::loss_opt_string(std::string vars){
    auto lambb = [this](std::string* v) -> bool   {return this -> has_string(v, "true");};
    auto lambi = [this](std::string* v) -> int    {return std::stoi(*v);}; 
    auto lambd = [this](std::string* v) -> double {return std::stod(*v);}; 
    auto lambv = [this](std::string* v) -> std::vector<double> {
        this -> replace(v, "{", ""); this -> replace(v, "}", ""); 
        std::vector<std::string> vcx = this -> split(*v, ",");
        std::vector<double> vo = {};
        for (size_t x(0); x < vcx.size(); ++x){this -> replace(&vcx[x], " ", "");}
        for (size_t x(0); x < vcx.size(); ++x){vo.push_back(std::stod(vcx[x]));}
        return vo; 
    }; 



    std::string _vars = this -> lower(&vars); 
    std::vector<std::string> vx = this -> split(_vars, "->"); 
    if (vx.size() != 2){
        this -> warning("Invalid parameter: " + vars + "\nExpected Syntax: ::(var -> val | var -> {list}");
        return;
    }
    this -> replace(&vx[0], " ", ""); 
    std::string name = vx[0]; 
    std::string val  = vx[1]; 
    if (name == "mean"      ){this -> lss_cfg.mean       = lambb(&val); return;}
    if (name == "sum"       ){this -> lss_cfg.sum        = lambb(&val); return;}
    if (name == "none"      ){this -> lss_cfg.sum        = lambb(&val); return;}
    if (name == "swap"      ){this -> lss_cfg.swap       = lambb(&val); return;}
    if (name == "full"      ){this -> lss_cfg.full       = lambb(&val); return;}
    if (name == "batch_mean"){this -> lss_cfg.batch_mean = lambb(&val); return;}
    if (name == "target"    ){this -> lss_cfg.target     = lambb(&val); return;} 
    if (name == "zero_inf"  ){this -> lss_cfg.zero_inf   = lambb(&val); return;}
    if (name == "ignore"    ){this -> lss_cfg.ignore     = lambi(&val); return;}
    if (name == "blank"     ){this -> lss_cfg.blank      = lambi(&val); return;}
    if (name == "margin"    ){this -> lss_cfg.margin     = lambd(&val); return;} 
    if (name == "beta"      ){this -> lss_cfg.beta       = lambd(&val); return;}
    if (name == "eps"       ){this -> lss_cfg.eps        = lambd(&val); return;}
    if (name == "smoothing" ){this -> lss_cfg.smoothing  = lambd(&val); return;}
    if (name == "weight"    ){this -> lss_cfg.weight     = lambv(&val); return;}
    this -> warning("Found invalid parameter: " + vars + " skipping"); 
}


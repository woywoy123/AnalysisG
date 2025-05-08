#include <templates/lossfx.h>

void lossfx::interpret(std::string* val){
    std::vector<std::string> vx = this -> split(*val, "::");
    this -> lss_cfg.fx = this -> loss_string(vx[0]); 
    if (vx.size() == 1){return;}
    this -> lss_cfg.defaults = false; 

    std::string params = vx[1];
    this -> replace(&params, "(", ""); 
    this -> replace(&params, ")", "");
    vx = this -> split(params, "|");
    for (size_t x(0); x < vx.size(); ++x){this -> loss_opt_string(vx[x]);}
}

void lossfx::build_fx_loss(torch::nn::BCELossImpl* lossfx_){
    if (lossfx_){return;}
    torch::nn::BCELossOptions opx; 
    _dress_reduction(&opx, &this -> lss_cfg);
    this -> m_bce = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults); 
} 

void lossfx::build_fx_loss(torch::nn::BCEWithLogitsLossImpl* lossfx_){
    if (lossfx_){return;}
    torch::nn::BCEWithLogitsLossOptions opx; 
    _dress_reduction(&opx, &this -> lss_cfg);
    this -> m_bce_with_logits = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults); 
}


void lossfx::build_fx_loss(torch::nn::CosineEmbeddingLossImpl* lossfx_){
    if (lossfx_){return;}
    torch::nn::CosineEmbeddingLossOptions opx;
    _dress_margin(&opx, &this -> lss_cfg);
    _dress_reduction(&opx, &this -> lss_cfg); 
    this -> m_cosine_embedding = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults); 
}


void lossfx::build_fx_loss(torch::nn::CrossEntropyLossImpl* lossfx_){
    if (lossfx_){return;}
    torch::nn::CrossEntropyLossOptions opx;
    _dress_reduction(&opx, &this -> lss_cfg);
    _dress_smoothing(&opx, &this -> lss_cfg); 
    _dress_ignore(&opx,    &this -> lss_cfg);  

    this -> m_cross_entropy = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults); 
} 

void lossfx::build_fx_loss(torch::nn::CTCLossImpl* lossfx_){
    if (lossfx_){return;}
    torch::nn::CTCLossOptions opx;
    _dress_reduction(&opx, &this -> lss_cfg);
    _dress_blank(&opx, &this -> lss_cfg); 
    _dress_zero(&opx, &this -> lss_cfg); 
    this -> m_ctc = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults); 
}


void lossfx::build_fx_loss(torch::nn::HingeEmbeddingLossImpl* lossfx_){
    if (lossfx_){return;}
    torch::nn::HingeEmbeddingLossOptions opx;
    _dress_margin(&opx, &this -> lss_cfg); 
    _dress_reduction(&opx, &this -> lss_cfg);
    this -> m_hinge_embedding = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults); 
}

void lossfx::build_fx_loss(torch::nn::HuberLossImpl* lossfx_){
    if (lossfx_){return;}
    torch::nn::HuberLossOptions opx;
    _dress_reduction(&opx, &this -> lss_cfg); 
    _dress_delta(&opx, &this -> lss_cfg);
    this -> m_huber = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults); 
}

void lossfx::build_fx_loss(torch::nn::KLDivLossImpl* lossfx_){
    if (lossfx_){return;}
    torch::nn::KLDivLossOptions opx;
    _dress_reduction(&opx, &this -> lss_cfg); 
    _dress_batch(&opx, &this -> lss_cfg); 
    _dress_target(&opx, &this -> lss_cfg);
    this -> m_kl_div = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults); 
}

void lossfx::build_fx_loss(torch::nn::L1LossImpl* lossfx_){
    if (lossfx_){return;}
    torch::nn::L1LossOptions opx;
    _dress_reduction(&opx, &this -> lss_cfg); 
    this -> m_l1 = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults);  
}

void lossfx::build_fx_loss(torch::nn::MarginRankingLossImpl* lossfx_){
    if (lossfx_){return;}
    torch::nn::MarginRankingLossOptions opx;
    _dress_reduction(&opx, &this -> lss_cfg); 
    _dress_margin(&opx, &this -> lss_cfg); 
    this -> m_margin_ranking = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults); 
}

void lossfx::build_fx_loss(torch::nn::MSELossImpl* lossfx_){
    if (lossfx_){return;}
    torch::nn::MSELossOptions opx;
    this -> m_mse = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults); 
}

void lossfx::build_fx_loss(torch::nn::MultiLabelMarginLossImpl* lossfx_){
    if (lossfx_){return;}
    torch::nn::MultiLabelMarginLossOptions opx;

    this -> m_multi_label_margin = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults); 
}

void lossfx::build_fx_loss(torch::nn::MultiLabelSoftMarginLossImpl* lossfx_){
    if (lossfx_){return;}
    torch::nn::MultiLabelSoftMarginLossOptions opx;

    this -> m_multi_label_soft_margin = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults); 
}

void lossfx::build_fx_loss(torch::nn::MultiMarginLossImpl* lossfx_){
    if (lossfx_){return;}
    torch::nn::MultiMarginLossOptions opx;
    this -> m_multi_margin = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults); 
}

void lossfx::build_fx_loss(torch::nn::NLLLossImpl* lossfx_){
    if (lossfx_){return;}
    torch::nn::NLLLossOptions opx;
    _dress_reduction(&opx, &this -> lss_cfg); 
    _dress_ignore(&opx, &this -> lss_cfg); 
    this -> m_nll = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults);  
}

void lossfx::build_fx_loss(torch::nn::PoissonNLLLossImpl* lossfx_){
    if (lossfx_){return;}
    torch::nn::PoissonNLLLossOptions opx;
    _dress_reduction(&opx, &this -> lss_cfg); 
    _dress_eps(&opx, &this -> lss_cfg); 
    _dress_full(&opx, &this -> lss_cfg); 
    this -> m_poisson_nll = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults); 
}

void lossfx::build_fx_loss(torch::nn::SmoothL1LossImpl* lossfx_){
    if (lossfx_){return;}
    torch::nn::SmoothL1LossOptions opx;
    _dress_reduction(&opx, &this -> lss_cfg); 
    _dress_beta(&opx, &this -> lss_cfg); 
    this -> m_smooth_l1 = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults); 
}

void lossfx::build_fx_loss(torch::nn::SoftMarginLossImpl*lossfx_){
    if (lossfx_){return;}
    torch::nn::SoftMarginLossOptions opx;
    _dress_reduction(&opx, &this -> lss_cfg); 
    this -> m_soft_margin = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults); 
}


void lossfx::build_fx_loss(torch::nn::TripletMarginLossImpl* lossfx_){
    if (lossfx_){return;}
    torch::nn::TripletMarginLossOptions opx;
    this -> m_triplet_margin = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults);  
}

void lossfx::build_fx_loss(torch::nn::TripletMarginWithDistanceLossImpl* lossfx_){
    if (lossfx_){return;}
    torch::nn::TripletMarginWithDistanceLossOptions opx;
    _dress_reduction(&opx, &this -> lss_cfg); 
    _dress_swap(&opx, &this -> lss_cfg); 
    _dress_margin(&opx, &this -> lss_cfg); 
    this -> m_triplet_margin_with_distance = generate_ops(lossfx_, &opx, this -> lss_cfg.defaults); 
}




torch::Tensor lossfx::loss(torch::Tensor* pred, torch::Tensor* truth){
    switch (this -> lss_cfg.fx){
        case loss_enum::bce                         : return this -> _fx_loss(this -> m_bce                         , pred, truth);  
        case loss_enum::bce_with_logits             : return this -> _fx_loss(this -> m_bce_with_logits             , pred, truth);  
        case loss_enum::cosine_embedding            : return this -> _fx_loss(this -> m_cosine_embedding            , pred, truth);  
        case loss_enum::cross_entropy               : return this -> _fx_loss(this -> m_cross_entropy               , pred, truth);  
        case loss_enum::ctc                         : return this -> _fx_loss(this -> m_ctc                         , pred, truth);  
        case loss_enum::hinge_embedding             : return this -> _fx_loss(this -> m_hinge_embedding             , pred, truth);  
        case loss_enum::huber                       : return this -> _fx_loss(this -> m_huber                       , pred, truth);  
        case loss_enum::kl_div                      : return this -> _fx_loss(this -> m_kl_div                      , pred, truth);  
        case loss_enum::l1                          : return this -> _fx_loss(this -> m_l1                          , pred, truth);  
        case loss_enum::margin_ranking              : return this -> _fx_loss(this -> m_margin_ranking              , pred, truth);  
        case loss_enum::mse                         : return this -> _fx_loss(this -> m_mse                         , pred, truth);  
        case loss_enum::multi_label_margin          : return this -> _fx_loss(this -> m_multi_label_margin          , pred, truth);  
        case loss_enum::multi_label_soft_margin     : return this -> _fx_loss(this -> m_multi_label_soft_margin     , pred, truth);  
        case loss_enum::multi_margin                : return this -> _fx_loss(this -> m_multi_margin                , pred, truth);  
        case loss_enum::nll                         : return this -> _fx_loss(this -> m_nll                         , pred, truth);  
        case loss_enum::poisson_nll                 : return this -> _fx_loss(this -> m_poisson_nll                 , pred, truth);  
        case loss_enum::smooth_l1                   : return this -> _fx_loss(this -> m_smooth_l1                   , pred, truth);  
        case loss_enum::soft_margin                 : return this -> _fx_loss(this -> m_soft_margin                 , pred, truth);  
        case loss_enum::triplet_margin              : return this -> _fx_loss(this -> m_triplet_margin              , pred, truth);  
        case loss_enum::triplet_margin_with_distance: return this -> _fx_loss(this -> m_triplet_margin_with_distance, pred, truth); 
        default: this -> warning("Invalid Loss Function!"); return torch::Tensor(); 
    }
}

torch::Tensor lossfx::loss(torch::Tensor* pred, torch::Tensor* truth, loss_enum lss){
    switch (lss){
        case loss_enum::bce                         : return this -> _fx_loss(this -> m_bce                         , pred, truth);  
        case loss_enum::bce_with_logits             : return this -> _fx_loss(this -> m_bce_with_logits             , pred, truth);  
        case loss_enum::cosine_embedding            : return this -> _fx_loss(this -> m_cosine_embedding            , pred, truth);  
        case loss_enum::cross_entropy               : return this -> _fx_loss(this -> m_cross_entropy               , pred, truth);  
        case loss_enum::ctc                         : return this -> _fx_loss(this -> m_ctc                         , pred, truth);  
        case loss_enum::hinge_embedding             : return this -> _fx_loss(this -> m_hinge_embedding             , pred, truth);  
        case loss_enum::huber                       : return this -> _fx_loss(this -> m_huber                       , pred, truth);  
        case loss_enum::kl_div                      : return this -> _fx_loss(this -> m_kl_div                      , pred, truth);  
        case loss_enum::l1                          : return this -> _fx_loss(this -> m_l1                          , pred, truth);  
        case loss_enum::margin_ranking              : return this -> _fx_loss(this -> m_margin_ranking              , pred, truth);  
        case loss_enum::mse                         : return this -> _fx_loss(this -> m_mse                         , pred, truth);  
        case loss_enum::multi_label_margin          : return this -> _fx_loss(this -> m_multi_label_margin          , pred, truth);  
        case loss_enum::multi_label_soft_margin     : return this -> _fx_loss(this -> m_multi_label_soft_margin     , pred, truth);  
        case loss_enum::multi_margin                : return this -> _fx_loss(this -> m_multi_margin                , pred, truth);  
        case loss_enum::nll                         : return this -> _fx_loss(this -> m_nll                         , pred, truth);  
        case loss_enum::poisson_nll                 : return this -> _fx_loss(this -> m_poisson_nll                 , pred, truth);  
        case loss_enum::smooth_l1                   : return this -> _fx_loss(this -> m_smooth_l1                   , pred, truth);  
        case loss_enum::soft_margin                 : return this -> _fx_loss(this -> m_soft_margin                 , pred, truth);  
        case loss_enum::triplet_margin              : return this -> _fx_loss(this -> m_triplet_margin              , pred, truth);  
        case loss_enum::triplet_margin_with_distance: return this -> _fx_loss(this -> m_triplet_margin_with_distance, pred, truth); 
        default: this -> warning("Invalid Loss Function!"); return torch::Tensor(); 
    }
}

torch::Tensor lossfx::_fx_loss(torch::nn::BCELossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    return lossfx_ -> forward(*pred, truth -> view({-1}).to(torch::kInt64)); 
}

torch::Tensor lossfx::_fx_loss(torch::nn::BCEWithLogitsLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    return lossfx_ -> forward(*pred, truth -> view({-1}).to(torch::kInt64)); 

}

torch::Tensor lossfx::_fx_loss(torch::nn::CosineEmbeddingLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    this -> failure("Not implemented! (modules/lossfx/cxx/loss_config.cxx)"); 
    return torch::Tensor(); 
}

torch::Tensor lossfx::_fx_loss(torch::nn::CrossEntropyLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    return lossfx_ -> forward(*pred, truth -> view({-1}).to(torch::kInt64)); 
}

torch::Tensor lossfx::_fx_loss(torch::nn::CTCLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    this -> failure("Not implemented! (modules/lossfx/cxx/loss_config.cxx)"); 
    return torch::Tensor(); //lossfx_ -> forward(*pred, truth -> view({-1}).to(torch::kInt64)); 

}

torch::Tensor lossfx::_fx_loss(torch::nn::HingeEmbeddingLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    return lossfx_ -> forward(*pred, truth -> view({-1}).to(torch::kInt64)); 

}

torch::Tensor lossfx::_fx_loss(torch::nn::HuberLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    return lossfx_ -> forward(*pred, truth -> view({-1}).to(torch::kInt64)); 

}

torch::Tensor lossfx::_fx_loss(torch::nn::KLDivLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    return lossfx_ -> forward(*pred, truth -> view({-1}).to(torch::kInt64)); 

}

torch::Tensor lossfx::_fx_loss(torch::nn::L1LossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    return lossfx_ -> forward(*pred, truth -> view({-1}).to(torch::kInt64)); 

}

torch::Tensor lossfx::_fx_loss(torch::nn::MarginRankingLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    this -> failure("Not implemented! (modules/lossfx/cxx/loss_config.cxx)"); 
    return torch::Tensor(); //lossfx_ -> forward(*pred, truth -> view({-1}).to(torch::kInt64));

}

torch::Tensor lossfx::_fx_loss(torch::nn::MSELossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    return lossfx_ -> forward(*pred, truth -> view({-1}).to(torch::kInt64));

}

torch::Tensor lossfx::_fx_loss(torch::nn::MultiLabelMarginLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    return lossfx_ -> forward(*pred, truth -> view({-1}).to(torch::kInt64));

}

torch::Tensor lossfx::_fx_loss(torch::nn::MultiLabelSoftMarginLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    return lossfx_ -> forward(*pred, truth -> view({-1}).to(torch::kInt64));

}

torch::Tensor lossfx::_fx_loss(torch::nn::MultiMarginLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    return lossfx_ -> forward(*pred, truth -> view({-1}).to(torch::kInt64));
}

torch::Tensor lossfx::_fx_loss(torch::nn::NLLLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    return lossfx_ -> forward(*pred, truth -> view({-1}).to(torch::kInt64));
}

torch::Tensor lossfx::_fx_loss(torch::nn::PoissonNLLLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    return lossfx_ -> forward(*pred, truth -> view({-1}).to(torch::kInt64));
}

torch::Tensor lossfx::_fx_loss(torch::nn::SmoothL1LossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    return lossfx_ -> forward(*pred, truth -> view({-1}).to(torch::kInt64));
}

torch::Tensor lossfx::_fx_loss(torch::nn::SoftMarginLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    return lossfx_ -> forward(*pred, truth -> view({-1}).to(torch::kInt64));
}

torch::Tensor lossfx::_fx_loss(torch::nn::TripletMarginLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    this -> failure("Not implemented! (modules/lossfx/cxx/loss_config.cxx)"); 
    return torch::Tensor(); //lossfx_ -> forward(*pred, truth -> view({-1}).to(torch::kInt64));
}

torch::Tensor lossfx::_fx_loss(torch::nn::TripletMarginWithDistanceLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth){
    this -> failure("Not implemented! (modules/lossfx/cxx/loss_config.cxx)"); 
    return torch::Tensor(); //lossfx_ -> forward(*pred, truth -> view({-1}).to(torch::kInt64));
}






#include <templates/lossfx.h>

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
        default: return torch::Tensor(); 
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






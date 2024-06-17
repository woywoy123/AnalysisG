#include <templates/model_template.h>

void model_template::set_output_features(
        std::map<std::string, std::string>* inpt, 
        std::map<std::string, std::tuple<torch::Tensor*, torch::nn::Module*, loss_enum>>* out_fx
){
    notification nx = notification(); 
    std::map<std::string, std::string>::iterator itx = inpt -> begin();
    for (; itx != inpt -> end(); ++itx){
        std::string o_fx = itx -> first; 
        std::string l_fx = itx -> second;

        torch::nn::Module* lss = nullptr; 
        loss_enum loss_type = loss_from_string(tools().lower(&l_fx));
        switch (loss_type){
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
        (*out_fx)[o_fx] = {nullptr, lss, loss_type}; 
        nx.success("Added loss function: " + l_fx + " for " + o_fx);
    }
}

torch::Tensor model_template::compute_loss(torch::Tensor* pred, std::tuple<torch::Tensor*, torch::nn::Module*, loss_enum>* lfx){
    torch::Tensor*  truth = std::get<0>(*lfx);
    torch::nn::Module* lx = std::get<1>(*lfx);
    switch (std::get<2>(*lfx)){
        case loss_enum::bce                         : return ((torch::nn::BCELossImpl*)(lx)) -> forward(*truth, *pred); 
        case loss_enum::bce_with_logits             : return ((torch::nn::BCEWithLogitsLossImpl*)(lx)) -> forward(*truth, *pred); 
        //case loss_enum::cosine_embedding            : return dynamic_cast<torch::nn::CosineEmbeddingLossImpl*>(lx) -> forward(*truth, *pred); 
        case loss_enum::cross_entropy               : return ((torch::nn::CrossEntropyLossImpl*)(lx)) -> forward(*pred, truth -> view({-1}).to(torch::kInt64)); 
        //case loss_enum::ctc                         : return dynamic_cast<torch::nn::CTCLossImpl*>(lx) -> forward(*truth, *pred); 
        case loss_enum::hinge_embedding             : return ((torch::nn::HingeEmbeddingLossImpl*)(lx)) -> forward(*truth, *pred); 
        case loss_enum::huber                       : return ((torch::nn::HuberLossImpl*)(lx)) -> forward(*truth, *pred); 
        case loss_enum::kl_div                      : return ((torch::nn::KLDivLossImpl*)(lx)) -> forward(*truth, *pred); 
        case loss_enum::l1                          : return ((torch::nn::L1LossImpl*)(lx)) -> forward(*truth, *pred); 
        //case loss_enum::margin_ranking              : return dynamic_cast<torch::nn::MarginRankingLossImpl*>(lx) -> forward(*truth, *pred); 
        case loss_enum::mse                         : return ((torch::nn::MSELossImpl*)(lx)) -> forward(*truth, *pred); 
        case loss_enum::multi_label_margin          : return ((torch::nn::MultiLabelMarginLossImpl*)(lx)) -> forward(*truth, *pred); 
        case loss_enum::multi_label_soft_margin     : return ((torch::nn::MultiLabelSoftMarginLossImpl*)(lx)) -> forward(*truth, *pred); 
        case loss_enum::multi_margin                : return ((torch::nn::MultiMarginLossImpl*)(lx)) -> forward(*truth, *pred); 
        case loss_enum::nll                         : return ((torch::nn::NLLLossImpl*)(lx)) -> forward(*truth, *pred); 
        case loss_enum::poisson_nll                 : return ((torch::nn::PoissonNLLLossImpl*)(lx)) -> forward(*truth, *pred); 
        case loss_enum::smooth_l1                   : return ((torch::nn::SmoothL1LossImpl*)(lx)) -> forward(*truth, *pred); 
        case loss_enum::soft_margin                 : return ((torch::nn::SoftMarginLossImpl*)(lx)) -> forward(*truth, *pred); 
        //case loss_enum::triplet_margin              : return dynamic_cast<torch::nn::TripletMarginLossImpl*>(lx) -> forward(*truth, *pred); 
        //case loss_enum::triplet_margin_with_distance: return dynamic_cast<torch::nn::TripletMarginWithDistanceLossImpl*>(lx) -> forward(*truth, *pred); 
        default: return *pred; 
    }
    return *pred; 
}



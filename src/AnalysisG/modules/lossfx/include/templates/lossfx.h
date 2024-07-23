#ifndef LOSSFX_H
#define LOSSFX_H
#include <map>
#include <vector>
#include <string>
#include <tools/tools.h>
#include <torch/torch.h>
#include <structs/model.h>
#include <templates/fx_enums.h>


class lossfx : public tools
{
    public:
        lossfx();
        ~lossfx();

        loss_enum loss_string(std::string name); 
        opt_enum optim_string(std::string name); 
        torch::Tensor loss(torch::Tensor* pred, torch::Tensor* truth, loss_enum lss); 
        void weight_init(torch::nn::Sequential* data, mlp_init method); 

        torch::optim::Optimizer* build_optimizer(optimizer_params_t* op, std::vector<torch::Tensor>* params); 
        bool build_loss_function(loss_enum lss); 

    private:
        void build_adam(optimizer_params_t* op, std::vector<torch::Tensor>* params); 
        void build_adagrad(optimizer_params_t* op, std::vector<torch::Tensor>* params); 
        void build_adamw(optimizer_params_t* op, std::vector<torch::Tensor>* params); 
        void build_lbfgs(optimizer_params_t* op, std::vector<torch::Tensor>* params); 
        void build_rmsprop(optimizer_params_t* op, std::vector<torch::Tensor>* params); 
        void build_sgd(optimizer_params_t* op, std::vector<torch::Tensor>* params); 

        // loss functions
        torch::Tensor _fx_loss(torch::nn::BCELossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   
        torch::Tensor _fx_loss(torch::nn::BCEWithLogitsLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   
        torch::Tensor _fx_loss(torch::nn::CosineEmbeddingLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   
        torch::Tensor _fx_loss(torch::nn::CrossEntropyLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   
        torch::Tensor _fx_loss(torch::nn::CTCLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   
        torch::Tensor _fx_loss(torch::nn::HingeEmbeddingLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   
        torch::Tensor _fx_loss(torch::nn::HuberLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   
        torch::Tensor _fx_loss(torch::nn::KLDivLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   
        torch::Tensor _fx_loss(torch::nn::L1LossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   
        torch::Tensor _fx_loss(torch::nn::MarginRankingLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   
        torch::Tensor _fx_loss(torch::nn::MSELossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   
        torch::Tensor _fx_loss(torch::nn::MultiLabelMarginLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   
        torch::Tensor _fx_loss(torch::nn::MultiLabelSoftMarginLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   
        torch::Tensor _fx_loss(torch::nn::MultiMarginLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   
        torch::Tensor _fx_loss(torch::nn::NLLLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   
        torch::Tensor _fx_loss(torch::nn::PoissonNLLLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   
        torch::Tensor _fx_loss(torch::nn::SmoothL1LossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   
        torch::Tensor _fx_loss(torch::nn::SoftMarginLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   
        torch::Tensor _fx_loss(torch::nn::TripletMarginLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   
        torch::Tensor _fx_loss(torch::nn::TripletMarginWithDistanceLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth);   


        // Optimizers 
        torch::optim::Adam*     m_adam    = nullptr; 
        torch::optim::Adagrad*  m_adagrad = nullptr;
        torch::optim::AdamW*    m_adamw   = nullptr; 
        torch::optim::LBFGS*    m_lbfgs   = nullptr; 
        torch::optim::RMSprop*  m_rmsprop = nullptr; 
        torch::optim::SGD*      m_sgd     = nullptr;  

        torch::nn::BCELossImpl*                       m_bce                          = nullptr;    
        torch::nn::BCEWithLogitsLossImpl*             m_bce_with_logits              = nullptr;    
        torch::nn::CosineEmbeddingLossImpl*           m_cosine_embedding             = nullptr;    
        torch::nn::CrossEntropyLossImpl*              m_cross_entropy                = nullptr;    
        torch::nn::CTCLossImpl*                       m_ctc                          = nullptr;    
        torch::nn::HingeEmbeddingLossImpl*            m_hinge_embedding              = nullptr;    
        torch::nn::HuberLossImpl*                     m_huber                        = nullptr;    
        torch::nn::KLDivLossImpl*                     m_kl_div                       = nullptr;    
        torch::nn::L1LossImpl*                        m_l1                           = nullptr;    
        torch::nn::MarginRankingLossImpl*             m_margin_ranking               = nullptr;    
        torch::nn::MSELossImpl*                       m_mse                          = nullptr;    
        torch::nn::MultiLabelMarginLossImpl*          m_multi_label_margin           = nullptr;    
        torch::nn::MultiLabelSoftMarginLossImpl*      m_multi_label_soft_margin      = nullptr;    
        torch::nn::MultiMarginLossImpl*               m_multi_margin                 = nullptr;    
        torch::nn::NLLLossImpl*                       m_nll                          = nullptr;    
        torch::nn::PoissonNLLLossImpl*                m_poisson_nll                  = nullptr;    
        torch::nn::SmoothL1LossImpl*                  m_smooth_l1                    = nullptr;    
        torch::nn::SoftMarginLossImpl*                m_soft_margin                  = nullptr;    
        torch::nn::TripletMarginLossImpl*             m_triplet_margin               = nullptr;    
        torch::nn::TripletMarginWithDistanceLossImpl* m_triplet_margin_with_distance = nullptr;    

}; 

#endif

#ifndef LOSSFX_H
#define LOSSFX_H

#include <map>
#include <vector>
#include <string>
#include <tools/tools.h>
#include <torch/torch.h>
#include <structs/enums.h>
#include <structs/optimizer.h>
#include <notification/notification.h>

template <typename g, typename go>
g* generate_ops(g* imp, go* opx, bool dx){
    if (dx){return new g();}
    return new g(*opx);
};

template <typename g>
void _dress_reduction(g* imx, loss_opt* params){
    if (params -> mean){imx -> reduction(torch::kMean); return;}
    if (params -> sum ){imx -> reduction(torch::kSum ); return;}
    if (params -> none){imx -> reduction(torch::kNone); return;}
};

template <typename g>
void _dress_batch(g* imx, loss_opt* params){
    if (!params -> batch_mean){return;}
    imx -> reduction(torch::kBatchMean);
}; 

template <typename g>
void _dress_ignore(g* imx, loss_opt* params){
    if (params -> ignore == 1000){return;}
    imx -> ignore_index(params -> ignore);
}; 

template <typename g>
void _dress_smoothing(g* imx, loss_opt* params){
    if (!params -> smoothing){return;}
    imx -> label_smoothing(params -> smoothing);
}; 

template <typename g>
void _dress_margin(g* imx, loss_opt* params){
    if (!params -> margin){return;}
    imx -> margin(params -> margin);
}; 

template <typename g>
void _dress_blank(g* imx, loss_opt* params){
    if (!params -> blank){return;}
    imx -> blank(params -> blank);
}; 

template <typename g>
void _dress_zero(g* imx, loss_opt* params){
    if (!params -> zero_inf){return;}
    imx -> zero_infinity(params -> zero_inf);
}; 

template <typename g>
void _dress_swap(g* imx, loss_opt* params){
    if (!params -> swap){return;}
    imx -> swap(params -> swap);
}; 

template <typename g>
void _dress_eps(g* imx, loss_opt* params){
    if (!params -> eps){return;}
    imx -> eps(params -> eps);
}; 

template <typename g>
void _dress_beta(g* imx, loss_opt* params){
    if (!params -> beta){return;}
    imx -> beta(params -> beta);
}; 

template <typename g>
void _dress_full(g* imx, loss_opt* params){
    if (!params -> full){return;}
    imx -> full(params -> full);
}; 

template <typename g>
void _dress_target(g* imx, loss_opt* params){
    if (!params -> target){return;}
    imx -> log_target(params -> target);
}; 

template <typename g>
void _dress_delta(g* imx, loss_opt* params){
    if (!params -> delta){return;}
    imx -> delta(params -> delta);
}; 


class lossfx: 
    public tools, 
    public notification
{
    public:
        lossfx();
        lossfx(std::string var, std::string enx);

        ~lossfx();

        loss_enum loss_string(std::string name); 
        opt_enum optim_string(std::string name); 
        scheduler_enum scheduler_string(std::string name);
        void loss_opt_string(std::string name); 

        torch::Tensor loss(torch::Tensor* pred, torch::Tensor* truth); 
        torch::Tensor loss(torch::Tensor* pred, torch::Tensor* truth, loss_enum lss); 

        void weight_init(torch::nn::Sequential* data, mlp_init method); 

        torch::optim::Optimizer* build_optimizer(optimizer_params_t* op, std::vector<torch::Tensor>* params); 
        void build_scheduler(optimizer_params_t* op, torch::optim::Optimizer* opx); 

        bool build_loss_function(loss_enum lss); 
        bool build_loss_function(); 

        void to(torch::TensorOptions*); 
        void step(); 
        std::string variable = ""; 
        loss_opt lss_cfg; 

    private:
        void interpret(std::string* ox); 

        void build_adam(optimizer_params_t* op, std::vector<torch::Tensor>* params); 
        void build_adagrad(optimizer_params_t* op, std::vector<torch::Tensor>* params); 
        void build_adamw(optimizer_params_t* op, std::vector<torch::Tensor>* params); 
        void build_lbfgs(optimizer_params_t* op, std::vector<torch::Tensor>* params); 
        void build_rmsprop(optimizer_params_t* op, std::vector<torch::Tensor>* params); 
        void build_sgd(optimizer_params_t* op, std::vector<torch::Tensor>* params); 

        void build_fx_loss(torch::nn::BCELossImpl* lossfx_);                      
        void build_fx_loss(torch::nn::BCEWithLogitsLossImpl* lossfx_);            
        void build_fx_loss(torch::nn::CosineEmbeddingLossImpl* lossfx_);          
        void build_fx_loss(torch::nn::CrossEntropyLossImpl* lossfx_);             
        void build_fx_loss(torch::nn::CTCLossImpl* lossfx_);                      
        void build_fx_loss(torch::nn::HingeEmbeddingLossImpl* lossfx_);           
        void build_fx_loss(torch::nn::HuberLossImpl* lossfx_);                    
        void build_fx_loss(torch::nn::KLDivLossImpl* lossfx_);                    
        void build_fx_loss(torch::nn::L1LossImpl* lossfx_);                       
        void build_fx_loss(torch::nn::MarginRankingLossImpl* lossfx_);            
        void build_fx_loss(torch::nn::MSELossImpl* lossfx_);                      
        void build_fx_loss(torch::nn::MultiLabelMarginLossImpl* lossfx_);         
        void build_fx_loss(torch::nn::MultiLabelSoftMarginLossImpl* lossfx_);     
        void build_fx_loss(torch::nn::MultiMarginLossImpl* lossfx_);              
        void build_fx_loss(torch::nn::NLLLossImpl* lossfx_);                      
        void build_fx_loss(torch::nn::PoissonNLLLossImpl* lossfx_);               
        void build_fx_loss(torch::nn::SmoothL1LossImpl* lossfx_);                 
        void build_fx_loss(torch::nn::SoftMarginLossImpl* lossfx_);                
        void build_fx_loss(torch::nn::TripletMarginLossImpl* lossfx_);             
        void build_fx_loss(torch::nn::TripletMarginWithDistanceLossImpl* lossfx_); 


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


        // ------------------ Optimizers ------------------ //
        torch::optim::Adam*     m_adam    = nullptr; 
        torch::optim::Adagrad*  m_adagrad = nullptr;
        torch::optim::AdamW*    m_adamw   = nullptr; 
        torch::optim::LBFGS*    m_lbfgs   = nullptr; 
        torch::optim::RMSprop*  m_rmsprop = nullptr; 
        torch::optim::SGD*      m_sgd     = nullptr;  

        // ------------ learning rate scheduler -------------- //
        torch::optim::StepLR*                     m_steplr = nullptr;
        torch::optim::ReduceLROnPlateauScheduler* m_rlp    = nullptr; 
        torch::optim::LRScheduler*                m_lrs    = nullptr; 

        // ----------------- loss functions --------------------------- //
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

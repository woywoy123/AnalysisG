#ifndef MODEL_ENUM_FUNCTIONS_H
#define MODEL_ENUM_FUNCTIONS_H

#include <string>
#include <structs/property.h>

struct optimizer_params_t {
    optimizer_params_t();

    std::string optimizer = ""; 

    cproperty<double, optimizer_params_t> lr;                           
    cproperty<double, optimizer_params_t> lr_decay;                     
    cproperty<double, optimizer_params_t> weight_decay;                 
    cproperty<double, optimizer_params_t> initial_accumulator_value;    
    cproperty<double, optimizer_params_t> eps;
    cproperty<double, optimizer_params_t> tolerance_grad;
    cproperty<double, optimizer_params_t> tolerance_change;
    cproperty<double, optimizer_params_t> alpha; 
    cproperty<double, optimizer_params_t> momentum;
    cproperty<double, optimizer_params_t> dampening; 

    cproperty<bool, optimizer_params_t> amsgrad; 
    cproperty<bool, optimizer_params_t> centered;
    cproperty<bool, optimizer_params_t> nesterov;

    cproperty<int, optimizer_params_t> max_iter; 
    cproperty<int, optimizer_params_t> max_eval; 
    cproperty<int, optimizer_params_t> history_size; 

    cproperty<std::tuple<float, float>, optimizer_params_t> betas; 
    cproperty<std::vector<float>, optimizer_params_t> beta_hack; 
    
    std::string scheduler = ""; 
    unsigned int step_size = 1; 
    double           gamma = 0.1;  
    

    bool m_lr                        = false; 
    bool m_lr_decay                  = false; 
    bool m_weight_decay              = false; 
    bool m_initial_accumulator_value = false; 
    bool m_eps                       = false; 
    bool m_betas                     = false; 
    bool m_amsgrad                   = false; 
    bool m_max_iter                  = false; 
    bool m_max_eval                  = false; 
    bool m_tolerance_grad            = false; 
    bool m_tolerance_change          = false; 
    bool m_history_size              = false; 
    bool m_alpha                     = false; 
    bool m_momentum                  = false;
    bool m_centered                  = false; 
    bool m_dampening                 = false; 
    bool m_nesterov                  = false;

    void static set_eps(double*, optimizer_params_t* obj);                         
    void static set_lr(double*, optimizer_params_t* obj);
    void static set_lr_decay(double*, optimizer_params_t* obj);
    void static set_weight_decay(double*, optimizer_params_t* obj); 
    void static set_initial_accumulator_value(double*, optimizer_params_t* obj);
    void static set_beta_hack(std::vector<float>* val, optimizer_params_t* obj);   
    void static set_betas(std::tuple<float, float>*, optimizer_params_t* obj);     
    void static set_amsgrad(bool*, optimizer_params_t* obj);                       
    void static set_max_iter(int*, optimizer_params_t* obj);                       
    void static set_max_eval(int*, optimizer_params_t* obj);                       
    void static set_tolerance_grad(double*, optimizer_params_t* obj);              
    void static set_tolerance_change(double*, optimizer_params_t* obj);            
    void static set_history_size(int*, optimizer_params_t* obj);                   
    void static set_centered(bool*, optimizer_params_t* obj);                      
    void static set_nesterov(bool*, optimizer_params_t* obj);                      
    void static set_alpha(double*, optimizer_params_t* obj);                       
    void static set_momentum(double*, optimizer_params_t* obj);                    
    void static set_dampening(double*, optimizer_params_t* obj);                   
}; 

#endif

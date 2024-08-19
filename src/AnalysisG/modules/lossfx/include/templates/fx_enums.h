#ifndef MODEL_ENUM_FUNCTIONS_H
#define MODEL_ENUM_FUNCTIONS_H

#include <string>
#include <structs/property.h>

struct optimizer_params_t {
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

    void operator()(){
        this -> lr.set_setter(this -> set_lr);
        this -> lr_decay.set_setter(this -> set_lr_decay);
        this -> weight_decay.set_setter(this -> set_weight_decay);
        this -> initial_accumulator_value.set_setter(this -> set_initial_accumulator_value);
        this -> eps.set_setter(this -> set_eps);
        this -> betas.set_setter(this -> set_betas);
        this -> amsgrad.set_setter(this -> set_amsgrad);
        this -> max_iter.set_setter(this -> set_max_iter);
        this -> max_eval.set_setter(this -> set_max_eval);
        this -> tolerance_grad.set_setter(this -> set_tolerance_grad);
        this -> tolerance_change.set_setter(this -> set_tolerance_change);
        this -> history_size.set_setter(this -> set_history_size);
        this -> centered.set_setter(this -> set_centered); 
        this -> nesterov.set_setter(this -> set_nesterov); 
        this -> alpha.set_setter(this -> set_alpha); 
        this -> momentum.set_setter(this -> set_momentum); 
        this -> dampening.set_setter(this -> set_dampening); 
        this -> beta_hack.set_setter(this -> set_beta_hack); 

        this -> lr.set_object(this);
        this -> lr_decay.set_object(this);
        this -> weight_decay.set_object(this);
        this -> initial_accumulator_value.set_object(this);
        this -> eps.set_object(this);
        this -> betas.set_object(this);
        this -> amsgrad.set_object(this);
        this -> max_iter.set_object(this);
        this -> max_eval.set_object(this);
        this -> tolerance_grad.set_object(this); 
        this -> tolerance_change.set_object(this);
        this -> history_size.set_object(this);
        this -> centered.set_object(this); 
        this -> nesterov.set_object(this); 
        this -> alpha.set_object(this); 
        this -> momentum.set_object(this); 
        this -> dampening.set_object(this); 
        this -> beta_hack.set_object(this); 
    }

    void static set_lr(double* val, optimizer_params_t* obj){obj -> m_lr = true;}
    void static set_lr_decay(double* val, optimizer_params_t* obj){obj -> m_lr_decay = true;}
    void static set_weight_decay(double* val, optimizer_params_t* obj){obj -> m_weight_decay = true;}
    void static set_initial_accumulator_value(double* val, optimizer_params_t* obj){obj -> m_initial_accumulator_value = true;}
    void static set_eps(double* val, optimizer_params_t* obj){obj -> m_eps = true;}
    void static set_betas(std::tuple<float, float>* val, optimizer_params_t* obj){obj -> m_betas = true;}
    void static set_amsgrad(bool* val, optimizer_params_t* obj){obj -> m_amsgrad = true;}
    void static set_max_iter(int* val, optimizer_params_t* obj){obj -> m_max_iter = true;}
    void static set_max_eval(int* val, optimizer_params_t* obj){obj -> m_max_eval = true;}
    void static set_tolerance_grad(double* val, optimizer_params_t* obj){obj -> m_tolerance_grad = true;}
    void static set_tolerance_change(double* val, optimizer_params_t* obj){obj -> m_tolerance_change = true;}
    void static set_history_size(int* val, optimizer_params_t* obj){obj -> m_history_size = true;}
    void static set_centered(bool* val, optimizer_params_t* obj){obj -> m_centered = true;}
    void static set_nesterov(bool* val, optimizer_params_t* obj){obj -> m_nesterov = true;}
    void static set_alpha(double* val, optimizer_params_t* obj){obj -> m_alpha = true;}
    void static set_momentum(double* val, optimizer_params_t* obj){obj -> m_momentum = true;}
    void static set_dampening(double* val, optimizer_params_t* obj){obj -> m_dampening = true;}
    void static set_beta_hack(std::vector<float>* val, optimizer_params_t* obj){
        std::tuple<float, float> x = {(*val)[0], (*val)[1]}; 
        obj -> betas = x; 
    }

}; 

#endif

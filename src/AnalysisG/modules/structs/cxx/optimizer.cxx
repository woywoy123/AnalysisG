#include <structs/optimizer.h>

optimizer_params_t::optimizer_params_t(){
    this -> betas.set_setter(this -> set_betas);
    this -> betas.set_object(this);

    this -> lr.set_setter(this -> set_lr);
    this -> lr.set_object(this);

    this -> lr_decay.set_setter(this -> set_lr_decay);
    this -> lr_decay.set_object(this);

    this -> weight_decay.set_setter(this -> set_weight_decay);
    this -> weight_decay.set_object(this);

    this -> initial_accumulator_value.set_setter(this -> set_initial_accumulator_value);
    this -> initial_accumulator_value.set_object(this);

    this -> eps.set_setter(this -> set_eps);
    this -> eps.set_object(this);

    this -> amsgrad.set_setter(this -> set_amsgrad);
    this -> amsgrad.set_object(this);

    this -> max_iter.set_setter(this -> set_max_iter);
    this -> max_iter.set_object(this);

    this -> max_eval.set_setter(this -> set_max_eval);
    this -> max_eval.set_object(this);

    this -> tolerance_grad.set_setter(this -> set_tolerance_grad);
    this -> tolerance_grad.set_object(this); 

    this -> tolerance_change.set_setter(this -> set_tolerance_change);
    this -> tolerance_change.set_object(this);

    this -> history_size.set_setter(this -> set_history_size);
    this -> history_size.set_object(this);

    this -> centered.set_setter(this -> set_centered); 
    this -> centered.set_object(this); 

    this -> nesterov.set_setter(this -> set_nesterov); 
    this -> nesterov.set_object(this); 

    this -> alpha.set_setter(this -> set_alpha); 
    this -> alpha.set_object(this); 

    this -> momentum.set_setter(this -> set_momentum); 
    this -> momentum.set_object(this); 

    this -> dampening.set_setter(this -> set_dampening); 
    this -> dampening.set_object(this); 

    this -> beta_hack.set_setter(this -> set_beta_hack); 
    this -> beta_hack.set_object(this); 
}

void optimizer_params_t::set_lr(double*, optimizer_params_t* obj){obj -> m_lr = true;}
void optimizer_params_t::set_lr_decay(double*, optimizer_params_t* obj){obj -> m_lr_decay = true;}
void optimizer_params_t::set_weight_decay(double*, optimizer_params_t* obj){obj -> m_weight_decay = true;}
void optimizer_params_t::set_initial_accumulator_value(double*, optimizer_params_t* obj){obj -> m_initial_accumulator_value = true;}
void optimizer_params_t::set_eps(double*, optimizer_params_t* obj){obj -> m_eps = true;}
void optimizer_params_t::set_betas(std::tuple<float, float>*, optimizer_params_t* obj){obj -> m_betas = true;}
void optimizer_params_t::set_amsgrad(bool*, optimizer_params_t* obj){obj -> m_amsgrad = true;}
void optimizer_params_t::set_max_iter(int*, optimizer_params_t* obj){obj -> m_max_iter = true;}
void optimizer_params_t::set_max_eval(int*, optimizer_params_t* obj){obj -> m_max_eval = true;}
void optimizer_params_t::set_tolerance_grad(double*, optimizer_params_t* obj){obj -> m_tolerance_grad = true;}
void optimizer_params_t::set_tolerance_change(double*, optimizer_params_t* obj){obj -> m_tolerance_change = true;}
void optimizer_params_t::set_history_size(int*, optimizer_params_t* obj){obj -> m_history_size = true;}
void optimizer_params_t::set_centered(bool*, optimizer_params_t* obj){obj -> m_centered = true;}
void optimizer_params_t::set_nesterov(bool*, optimizer_params_t* obj){obj -> m_nesterov = true;}
void optimizer_params_t::set_alpha(double*, optimizer_params_t* obj){obj -> m_alpha = true;}
void optimizer_params_t::set_momentum(double*, optimizer_params_t* obj){obj -> m_momentum = true;}
void optimizer_params_t::set_dampening(double*, optimizer_params_t* obj){obj -> m_dampening = true;}
void optimizer_params_t::set_beta_hack(std::vector<float>* val, optimizer_params_t* obj){
    std::tuple<float, float> x = {(*val)[0], (*val)[1]}; 
    obj -> betas = x; 
}



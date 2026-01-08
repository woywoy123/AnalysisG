/**
 * @file optimizer.h
 * @brief Defines the optimizer_params_t structure for configuring model optimizers.
 *
 * This file contains the declaration of the `optimizer_params_t` structure, which
 * holds all configuration parameters for PyTorch optimizers used during model training.
 * It supports various optimizer types including Adam, SGD, Adagrad, and LBFGS.
 */

#ifndef MODEL_ENUM_FUNCTIONS_H
#define MODEL_ENUM_FUNCTIONS_H

#include <string>
#include <structs/property.h>

/**
 * @struct optimizer_params_t
 * @brief Configuration parameters for PyTorch optimizers.
 *
 * This structure contains all the parameters needed to configure different
 * optimizer types in PyTorch. Parameters are exposed as cproperty objects
 * for seamless Python integration via Cython.
 *
 * @section optimizer_types Supported Optimizer Types
 *
 * Set the `optimizer` member to one of:
 * - "Adam": Adam optimizer
 * - "SGD": Stochastic Gradient Descent
 * - "Adagrad": Adaptive gradient
 * - "LBFGS": Limited-memory BFGS
 * - "RMSprop": RMSprop optimizer
 *
 * @section optimizer_example Usage Example
 *
 * ```cpp
 * optimizer_params_t params;
 * params.optimizer = "Adam";
 * params.lr = 0.001;
 * params.weight_decay = 1e-5;
 * params.betas = {0.9, 0.999};
 * params.amsgrad = true;
 * params();  // Initialize property callbacks
 *
 * // Use with model
 * analysis ana;
 * ana.add_model(new MyModel(), &params, "training_run");
 * ```
 *
 * @section optimizer_adam Adam Parameters
 *
 * For Adam optimizer, configure:
 * - `lr`: Learning rate (required)
 * - `betas`: Tuple of (beta1, beta2) for momentum terms
 * - `eps`: Epsilon for numerical stability
 * - `weight_decay`: L2 penalty coefficient
 * - `amsgrad`: Whether to use AMSGrad variant
 *
 * @section optimizer_sgd SGD Parameters
 *
 * For SGD optimizer, configure:
 * - `lr`: Learning rate (required)
 * - `momentum`: Momentum factor
 * - `weight_decay`: L2 penalty coefficient
 * - `dampening`: Dampening for momentum
 * - `nesterov`: Whether to enable Nesterov momentum
 *
 * @section optimizer_lbfgs LBFGS Parameters
 *
 * For LBFGS optimizer, configure:
 * - `lr`: Learning rate (required)
 * - `max_iter`: Maximum iterations per optimization step
 * - `max_eval`: Maximum function evaluations per step
 * - `tolerance_grad`: Termination tolerance on gradient norm
 * - `tolerance_change`: Termination tolerance on function value change
 * - `history_size`: Number of previous gradients to store
 */
struct optimizer_params_t {
    std::string optimizer = "";  ///< Optimizer type ("Adam", "SGD", "Adagrad", "LBFGS", "RMSprop").

    cproperty<double, optimizer_params_t> lr;                           ///< Learning rate.
    cproperty<double, optimizer_params_t> lr_decay;                     ///< Learning rate decay factor.
    cproperty<double, optimizer_params_t> weight_decay;                 ///< L2 regularization coefficient.
    cproperty<double, optimizer_params_t> initial_accumulator_value;    ///< Initial accumulator value (Adagrad).
    cproperty<double, optimizer_params_t> eps;                          ///< Epsilon for numerical stability.
    cproperty<double, optimizer_params_t> tolerance_grad;               ///< Gradient tolerance (LBFGS).
    cproperty<double, optimizer_params_t> tolerance_change;             ///< Function value change tolerance (LBFGS).
    cproperty<double, optimizer_params_t> alpha;                        ///< Smoothing constant (RMSprop).
    cproperty<double, optimizer_params_t> momentum;                     ///< Momentum factor (SGD, RMSprop).
    cproperty<double, optimizer_params_t> dampening;                    ///< Dampening for momentum (SGD).

    cproperty<bool, optimizer_params_t> amsgrad;                        ///< Use AMSGrad variant (Adam).
    cproperty<bool, optimizer_params_t> centered;                       ///< Compute centered RMSprop.
    cproperty<bool, optimizer_params_t> nesterov;                       ///< Use Nesterov momentum (SGD).

    cproperty<int, optimizer_params_t> max_iter;                        ///< Maximum iterations per step (LBFGS).
    cproperty<int, optimizer_params_t> max_eval;                        ///< Maximum evaluations per step (LBFGS).
    cproperty<int, optimizer_params_t> history_size;                    ///< History size for LBFGS.

    cproperty<std::tuple<float, float>, optimizer_params_t> betas;      ///< Beta coefficients (beta1, beta2) for Adam.
    cproperty<std::vector<float>, optimizer_params_t> beta_hack;        ///< Alternative beta specification as vector.

    // Flags indicating which parameters have been set
    bool m_lr                        = false;  ///< Flag: lr has been set.
    bool m_lr_decay                  = false;  ///< Flag: lr_decay has been set.
    bool m_weight_decay              = false;  ///< Flag: weight_decay has been set.
    bool m_initial_accumulator_value = false;  ///< Flag: initial_accumulator_value has been set.
    bool m_eps                       = false;  ///< Flag: eps has been set.
    bool m_betas                     = false;  ///< Flag: betas has been set.
    bool m_amsgrad                   = false;  ///< Flag: amsgrad has been set.
    bool m_max_iter                  = false;  ///< Flag: max_iter has been set.
    bool m_max_eval                  = false;  ///< Flag: max_eval has been set.
    bool m_tolerance_grad            = false;  ///< Flag: tolerance_grad has been set.
    bool m_tolerance_change          = false;  ///< Flag: tolerance_change has been set.
    bool m_history_size              = false;  ///< Flag: history_size has been set.
    bool m_alpha                     = false;  ///< Flag: alpha has been set.
    bool m_momentum                  = false;  ///< Flag: momentum has been set.
    bool m_centered                  = false;  ///< Flag: centered has been set.
    bool m_dampening                 = false;  ///< Flag: dampening has been set.
    bool m_nesterov                  = false;  ///< Flag: nesterov has been set.

    /**
     * @brief Initializes property callbacks.
     * Must be called after construction to set up getters/setters.
     */
    void operator()(); 
    
    // Setter functions for cproperty callbacks
    void static set_eps(double*, optimizer_params_t* obj);                         ///< Setter for eps.
    void static set_lr(double*, optimizer_params_t* obj);                          ///< Setter for lr.
    void static set_lr_decay(double*, optimizer_params_t* obj);                    ///< Setter for lr_decay.
    void static set_weight_decay(double*, optimizer_params_t* obj);                ///< Setter for weight_decay.
    void static set_initial_accumulator_value(double*, optimizer_params_t* obj);   ///< Setter for initial_accumulator_value.
    void static set_beta_hack(std::vector<float>* val, optimizer_params_t* obj);   ///< Setter for beta_hack.
    void static set_betas(std::tuple<float, float>*, optimizer_params_t* obj);     ///< Setter for betas.
    void static set_amsgrad(bool*, optimizer_params_t* obj);                       ///< Setter for amsgrad.
    void static set_max_iter(int*, optimizer_params_t* obj);                       ///< Setter for max_iter.
    void static set_max_eval(int*, optimizer_params_t* obj);                       ///< Setter for max_eval.
    void static set_tolerance_grad(double*, optimizer_params_t* obj);              ///< Setter for tolerance_grad.
    void static set_tolerance_change(double*, optimizer_params_t* obj);            ///< Setter for tolerance_change.
    void static set_history_size(int*, optimizer_params_t* obj);                   ///< Setter for history_size.
    void static set_centered(bool*, optimizer_params_t* obj);                      ///< Setter for centered.
    void static set_nesterov(bool*, optimizer_params_t* obj);                      ///< Setter for nesterov.
    void static set_alpha(double*, optimizer_params_t* obj);                       ///< Setter for alpha.
    void static set_momentum(double*, optimizer_params_t* obj);                    ///< Setter for momentum.
    void static set_dampening(double*, optimizer_params_t* obj);                   ///< Setter for dampening.
}; 

#endif

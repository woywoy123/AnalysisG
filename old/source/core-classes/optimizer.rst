The Optimizer Configuration Class
---------------------------------

This class is used to configure the model training parameters.
The class closely follows the `PyTorch` interface parameters.

.. py:class:: OptimizerConfig

   .. py:attribute:: Optimizer
      :type: str
      :value: adam, adagrad, adamw, lbfgs, rmsprop, sgd

      Specifies the optimizer to use.
    
   .. py:attribute:: lr
      :type: float

      Specifies the learning rate of the optimizer.
      
   .. py:attribute:: lr_decay
      :type: float
      
      Specifies the learning rate decay.

   .. py:attribute:: weight_decay
      :type: float

      Specifies the decay rate of the weights.

   .. py:attribute:: initial_accumulator_value

   .. py:attribute:: eps

   .. py:attribute:: tolerance_grad

   .. py:attribute:: tolerance_change

   .. py:attribute:: alpha

   .. py:attribute:: momentum

   .. py:attribute:: dampening

   .. py:attribute:: amsgrad

   .. py:attribute:: centered

   .. py:attribute:: nesterov

   .. py:attribute:: max_iter

   .. py:attribute:: max_eval

   .. py:attribute:: history_size

   .. py:attribute:: betas


The Optimizer Configuration Class in C++
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The point of the Python interface is to control the workflow of the Analysis framework, however, some of the loss functions may require additional tuning or modifications.
As such, the Optimizer class is given below with its class members.


.. code:: C++

   #include <templates/fx_enums.h>
   #include <templates/lossfx.h>

.. cpp:enum:: opt_enum

    adam
    adagrad
    adamw
    lbfgs 
    rmsprop
    sgd
    invalid_optimizer

.. cpp:enum:: mlp_init 

    uniform
    normal 
    xavier_normal
    xavier_uniform
    kaiming_uniform
    kaiming_normal

.. cpp:enum:: loss_enum

    bce
    bce_with_logits
    cosine_embedding 
    cross_entropy 
    ctc 
    hinge_embedding 
    huber 
    kl_div 
    l1 
    margin_ranking 
    mse 
    multi_label_margin 
    multi_label_soft_margin 
    multi_margin 
    nll 
    poisson_nll 
    smooth_l1 
    soft_margin 
    triplet_margin 
    triplet_margin_with_distance
    invalid_loss

.. cpp:enum:: graph_enum

    data_graph
    data_node
    data_edge
    truth_graph
    truth_node 
    truth_edge

.. cpp:struct:: optimizer_params_t

   .. cpp:var:: std::string optimizer 

      Specifies the optimizer by string name..

   .. cpp:var:: cproperty<double, optimizer_params_t> lr

   .. cpp:var:: cproperty<double, optimizer_params_t> lr_decay

   .. cpp:var:: cproperty<double, optimizer_params_t> weight_decay

   .. cpp:var:: cproperty<double, optimizer_params_t> initial_accumulator_value

   .. cpp:var:: cproperty<double, optimizer_params_t> eps

   .. cpp:var:: cproperty<double, optimizer_params_t> tolerance_grad

   .. cpp:var:: cproperty<double, optimizer_params_t> tolerance_change

   .. cpp:var:: cproperty<double, optimizer_params_t> alpha

   .. cpp:var:: cproperty<double, optimizer_params_t> momentum

   .. cpp:var:: cproperty<double, optimizer_params_t> dampening

   .. cpp:var:: cproperty<bool, optimizer_params_t> amsgrad

   .. cpp:var:: cproperty<bool, optimizer_params_t> centered

   .. cpp:var:: cproperty<bool, optimizer_params_t> nesterov

   .. cpp:var:: cproperty<int, optimizer_params_t> max_iter

   .. cpp:var:: cproperty<int, optimizer_params_t> max_eval

   .. cpp:var:: cproperty<int, optimizer_params_t> history_size

   .. cpp:var:: cproperty<std::tuple<float, float>, optimizer_params_t> betas

   .. cpp:var:: cproperty<std::vector<float>, optimizer_params_t> beta_hack

   .. cpp:var:: bool m_lr                        

   .. cpp:var:: bool m_lr_decay                  

   .. cpp:var:: bool m_weight_decay              

   .. cpp:var:: bool m_initial_accumulator_value 

   .. cpp:var:: bool m_eps                       

   .. cpp:var:: bool m_betas                     

   .. cpp:var:: bool m_amsgrad                   

   .. cpp:var:: bool m_max_iter                  

   .. cpp:var:: bool m_max_eval                  

   .. cpp:var:: bool m_tolerance_grad            

   .. cpp:var:: bool m_tolerance_change          

   .. cpp:var:: bool m_history_size              

   .. cpp:var:: bool m_alpha                     

   .. cpp:var:: bool m_momentum                  

   .. cpp:var:: bool m_centered                  

   .. cpp:var:: bool m_dampening                 

   .. cpp:var:: bool m_nesterov                  


.. cpp:class:: lossfx: public tools

   .. cpp:function:: loss_enum loss_string(std::string name)

      Maps the input loss function string name to the loss_enum.

   .. cpp:function:: opt_enum optim_string(std::string name)

      Maps the input optimizer string name to the opt_enum.

   .. cpp:function:: torch::Tensor loss(torch::Tensor* pred, torch::Tensor* truth, loss_enum lss)

      Computes the loss between the prediction and the underlying truth tensors, given the loss_enum.
      Not all functions have been fully implemented, but can be easily defined under (src/AnalysisG/modules/lossfx/cxx/loss_config.cxx).

   .. cpp:function:: void weight_init(torch::nn::Sequential* data, mlp_init method)

      Specifies how the MLP weights should be initialized.

   .. cpp:function:: torch::optim::Optimizer* build_optimizer(optimizer_params_t* op, std::vector<torch::Tensor>* params)

      Constructs the optimizer with specified parameters.

   .. cpp:function:: bool build_loss_function(loss_enum lss)

      Constructs the loss function using the enum value.

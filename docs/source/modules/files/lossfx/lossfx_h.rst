lossfx.h
========

**File Path**: ``modules/lossfx/include/templates/lossfx.h``

**File Type**: H (Header)

**Lines**: 227

Dependencies
------------

**Includes**:

- ``map``
- ``notification/notification.h``
- ``string``
- ``structs/enums.h``
- ``structs/optimizer.h``
- ``tools/tools.h``
- ``torch/torch.h``
- ``vector``

Classes
-------

``lossfx``
~~~~~~~~~~

**Inherits from**: ``tools, 
    public notification``

**Methods**:

- ``loss_enum loss_string(std::string name)``
- ``opt_enum optim_string(std::string name)``
- ``scheduler_enum scheduler_string(std::string name)``
- ``void loss_opt_string(std::string name)``
- ``Tensor loss(torch::Tensor* pred, torch::Tensor* truth)``
- ``Tensor loss(torch::Tensor* pred, torch::Tensor* truth, loss_en...)``
- ``void weight_init(torch::nn::Sequential* data, mlp_init method)``
- ``void build_scheduler(optimizer_params_t* op, torch::optim::Optimizer* o...)``
- ``bool build_loss_function(loss_enum lss)``
- ``bool build_loss_function()``

Functions
---------

``void _dress_reduction(g* imx, loss_opt* params)``

``void _dress_batch(g* imx, loss_opt* params)``

``void _dress_ignore(g* imx, loss_opt* params)``

``void _dress_smoothing(g* imx, loss_opt* params)``

``void _dress_margin(g* imx, loss_opt* params)``

``void _dress_blank(g* imx, loss_opt* params)``

``void _dress_zero(g* imx, loss_opt* params)``

``void _dress_swap(g* imx, loss_opt* params)``

``void _dress_eps(g* imx, loss_opt* params)``

``void _dress_beta(g* imx, loss_opt* params)``

``void _dress_full(g* imx, loss_opt* params)``

``void _dress_target(g* imx, loss_opt* params)``

``void _dress_delta(g* imx, loss_opt* params)``


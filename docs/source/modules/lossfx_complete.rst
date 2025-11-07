====================================================================================================
Complete Loss Functions and Optimizers Documentation
====================================================================================================

This document provides **exhaustive documentation** for all loss functions, optimizers, and
learning rate schedulers in the AnalysisG framework.

.. contents::
   :local:
   :depth: 3

Overview
====================================================================================================

The ``lossfx`` module (``modules/lossfx/``) provides a unified interface to PyTorch's loss functions
and optimizers. It supports **20 loss functions**, **6 optimizers**, and **2 learning rate schedulers**.

**Location**: ``src/AnalysisG/modules/lossfx/include/templates/lossfx.h``

**Dependencies**:
- PyTorch (torch::nn, torch::optim)
- notification (progress reporting)
- tools (utility functions)
- structs/enums (loss_enum, opt_enum, scheduler_enum)
- structs/optimizer (optimizer_params_t, loss_opt)

Class Definition
====================================================================================================

.. code-block:: cpp

   class lossfx: 
       public notification, 
       public tools
   {
       public:
           lossfx();
           lossfx(std::string var, std::string enx);
           ~lossfx();
           
           // Loss function interface
           torch::Tensor loss(torch::Tensor* pred, torch::Tensor* truth); 
           torch::Tensor loss(torch::Tensor* pred, torch::Tensor* truth, loss_enum lss);
           
           // Optimizer and scheduler builders
           torch::optim::Optimizer* build_optimizer(optimizer_params_t* op, std::vector<torch::Tensor>* params); 
           void build_scheduler(optimizer_params_t* op, torch::optim::Optimizer* opx); 
           bool build_loss_function(loss_enum lss); 
           bool build_loss_function();
           
           // Utility methods
           loss_enum loss_string(std::string name); 
           opt_enum optim_string(std::string name); 
           scheduler_enum scheduler_string(std::string name);
           void loss_opt_string(std::string name); 
           void weight_init(torch::nn::Sequential* data, mlp_init method); 
           void to(torch::TensorOptions*); 
           void step(); 
           
           // Public members
           std::string variable = ""; 
           loss_opt lss_cfg;
   };

All 20 Loss Functions
====================================================================================================

The framework supports all PyTorch loss functions with full configuration control:

1. BCELoss (Binary Cross Entropy Loss)
----------------------------------------------------------------------------------------------------

**Formula**: L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

**Use Cases**: Binary classification problems

**Parameters** (via ``loss_opt``):
- ``reduction``: "mean" (default), "sum", or "none"

**Configuration**:

.. code-block:: cpp

   loss_opt cfg;
   cfg.mean = true;  // Use mean reduction
   lossfx loss_fn;
   loss_fn.lss_cfg = cfg;
   loss_fn.build_loss_function(loss_enum::bce);

**PyTorch Member**: ``torch::nn::BCELossImpl* m_bce``

2. BCEWithLogitsLoss 
----------------------------------------------------------------------------------------------------

**Formula**: L = -[y·log(σ(x)) + (1-y)·log(1-σ(x))] where σ is sigmoid

**Use Cases**: Binary classification with raw logits (more numerically stable than BCELoss)

**Parameters**:
- ``reduction``: "mean", "sum", or "none"

**Advantage**: Combines sigmoid + BCE for numerical stability

**PyTorch Member**: ``torch::nn::BCEWithLogitsLossImpl* m_bce_with_logits``

3. CosineEmbeddingLoss
----------------------------------------------------------------------------------------------------

**Formula**: L = 1 - cos(x₁, x₂) if y=1, max(0, cos(x₁, x₂) - margin) if y=-1

**Use Cases**: Learning embeddings, similarity learning

**Parameters**:
- ``margin``: Margin for dissimilar pairs (default: 0)
- ``reduction``: "mean", "sum", or "none"

**Configuration**:

.. code-block:: cpp

   loss_opt cfg;
   cfg.margin = 0.5;  // Set margin for dissimilar pairs
   cfg.mean = true;

**PyTorch Member**: ``torch::nn::CosineEmbeddingLossImpl* m_cosine_embedding``

4. CrossEntropyLoss
----------------------------------------------------------------------------------------------------

**Formula**: L = -log(exp(xᵢ) / Σⱼexp(xⱼ)) where i is the target class

**Use Cases**: Multi-class classification (most common)

**Parameters**:
- ``reduction``: "mean", "sum", or "none"
- ``ignore``: Index to ignore (default: 1000, disabled)
- ``smoothing``: Label smoothing factor (default: 0)

**Configuration**:

.. code-block:: cpp

   loss_opt cfg;
   cfg.mean = true;
   cfg.ignore = 0;  // Ignore class 0 (e.g., padding)
   cfg.smoothing = 0.1;  // 10% label smoothing

**PyTorch Member**: ``torch::nn::CrossEntropyLossImpl* m_cross_entropy``

5. CTCLoss (Connectionist Temporal Classification Loss)
----------------------------------------------------------------------------------------------------

**Formula**: Sum over all valid alignments of input-output sequences

**Use Cases**: Sequence-to-sequence problems (speech recognition, OCR)

**Parameters**:
- ``blank``: Blank label index (default: 0)
- ``zero_inf``: Replace infinity with zero (default: false)
- ``reduction``: "mean", "sum", or "none"

**Configuration**:

.. code-block:: cpp

   loss_opt cfg;
   cfg.blank = 0;  // Blank token index
   cfg.zero_inf = true;  // Replace inf with 0
   cfg.mean = true;

**PyTorch Member**: ``torch::nn::CTCLossImpl* m_ctc``

6. HingeEmbeddingLoss
----------------------------------------------------------------------------------------------------

**Formula**: L = x if y=1, max(0, margin - x) if y=-1

**Use Cases**: Binary classification with margin

**Parameters**:
- ``margin``: Margin for negative examples (default: 1.0)
- ``reduction``: "mean", "sum", or "none"

**PyTorch Member**: ``torch::nn::HingeEmbeddingLossImpl* m_hinge_embedding``

7. HuberLoss
----------------------------------------------------------------------------------------------------

**Formula**: L = 0.5·(x-y)² if |x-y| < δ, δ·(|x-y| - 0.5·δ) otherwise

**Use Cases**: Robust regression (less sensitive to outliers than MSE)

**Parameters**:
- ``delta``: Threshold for quadratic vs. linear (default: 1.0)
- ``reduction``: "mean", "sum", or "none"

**Configuration**:

.. code-block:: cpp

   loss_opt cfg;
   cfg.delta = 1.0;  // Transition point
   cfg.mean = true;

**PyTorch Member**: ``torch::nn::HuberLossImpl* m_huber``

8. KLDivLoss (Kullback-Leibler Divergence Loss)
----------------------------------------------------------------------------------------------------

**Formula**: L = Σᵢ yᵢ·(log(yᵢ) - xᵢ)

**Use Cases**: Comparing probability distributions, distillation

**Parameters**:
- ``reduction``: "batchmean" (default), "mean", "sum", or "none"
- ``target``: Log target mode (default: false)

**Configuration**:

.. code-block:: cpp

   loss_opt cfg;
   cfg.batch_mean = true;  // Use batchmean reduction
   cfg.target = false;  // Input is log probabilities

**PyTorch Member**: ``torch::nn::KLDivLossImpl* m_kl_div``

9. L1Loss (Mean Absolute Error)
----------------------------------------------------------------------------------------------------

**Formula**: L = |x - y|

**Use Cases**: Regression, less sensitive to outliers

**Parameters**:
- ``reduction``: "mean", "sum", or "none"

**PyTorch Member**: ``torch::nn::L1LossImpl* m_l1``

10. MarginRankingLoss
----------------------------------------------------------------------------------------------------

**Formula**: L = max(0, -y·(x₁ - x₂) + margin)

**Use Cases**: Ranking problems, learning to rank

**Parameters**:
- ``margin``: Margin between rankings (default: 0)
- ``reduction``: "mean", "sum", or "none"

**PyTorch Member**: ``torch::nn::MarginRankingLossImpl* m_margin_ranking``

11. MSELoss (Mean Squared Error)
----------------------------------------------------------------------------------------------------

**Formula**: L = (x - y)²

**Use Cases**: Regression (most common)

**Parameters**:
- ``reduction``: "mean", "sum", or "none"

**PyTorch Member**: ``torch::nn::MSELossImpl* m_mse``

12. MultiLabelMarginLoss
----------------------------------------------------------------------------------------------------

**Formula**: Multi-label classification with margin

**Use Cases**: Multi-label classification problems

**Parameters**:
- ``reduction``: "mean", "sum", or "none"

**PyTorch Member**: ``torch::nn::MultiLabelMarginLossImpl* m_multi_label_margin``

13. MultiLabelSoftMarginLoss
----------------------------------------------------------------------------------------------------

**Formula**: Soft margin loss for multi-label classification

**Use Cases**: Multi-label classification with sigmoid

**Parameters**:
- ``reduction``: "mean", "sum", or "none"

**PyTorch Member**: ``torch::nn::MultiLabelSoftMarginLossImpl* m_multi_label_soft_margin``

14. MultiMarginLoss
----------------------------------------------------------------------------------------------------

**Formula**: Multi-class margin loss

**Use Cases**: Multi-class classification with margin

**Parameters**:
- ``margin``: Margin value (default: 1.0)
- ``reduction``: "mean", "sum", or "none"

**PyTorch Member**: ``torch::nn::MultiMarginLossImpl* m_multi_margin``

15. NLLLoss (Negative Log Likelihood Loss)
----------------------------------------------------------------------------------------------------

**Formula**: L = -xᵢ where i is the target class

**Use Cases**: Multi-class classification with log probabilities

**Parameters**:
- ``reduction``: "mean", "sum", or "none"
- ``ignore``: Index to ignore (default: 1000)

**PyTorch Member**: ``torch::nn::NLLLossImpl* m_nll``

16. PoissonNLLLoss
----------------------------------------------------------------------------------------------------

**Formula**: L = exp(x) - y·x

**Use Cases**: Poisson regression, count data

**Parameters**:
- ``reduction``: "mean", "sum", or "none"
- ``full``: Compute full loss including constant term (default: false)
- ``eps``: Small value to avoid log(0) (default: 1e-8)

**Configuration**:

.. code-block:: cpp

   loss_opt cfg;
   cfg.full = true;  // Include constant term
   cfg.eps = 1e-8;

**PyTorch Member**: ``torch::nn::PoissonNLLLossImpl* m_poisson_nll``

17. SmoothL1Loss (Huber Loss variant)
----------------------------------------------------------------------------------------------------

**Formula**: L = 0.5·(x-y)² if |x-y| < β, |x-y| - 0.5·β otherwise

**Use Cases**: Regression, object detection (used in Faster R-CNN)

**Parameters**:
- ``beta``: Threshold value (default: 1.0)
- ``reduction``: "mean", "sum", or "none"

**Configuration**:

.. code-block:: cpp

   loss_opt cfg;
   cfg.beta = 1.0;
   cfg.mean = true;

**PyTorch Member**: ``torch::nn::SmoothL1LossImpl* m_smooth_l1``

18. SoftMarginLoss
----------------------------------------------------------------------------------------------------

**Formula**: L = log(1 + exp(-y·x))

**Use Cases**: Binary classification with soft margin

**Parameters**:
- ``reduction``: "mean", "sum", or "none"

**PyTorch Member**: ``torch::nn::SoftMarginLossImpl* m_soft_margin``

19. TripletMarginLoss
----------------------------------------------------------------------------------------------------

**Formula**: L = max(0, ||a - p||₂ - ||a - n||₂ + margin)

**Use Cases**: Metric learning, face recognition, siamese networks

**Parameters**:
- ``margin``: Margin between positive and negative pairs (default: 1.0)
- ``swap``: Use distance swap (default: false)
- ``reduction``: "mean", "sum", or "none"

**Configuration**:

.. code-block:: cpp

   loss_opt cfg;
   cfg.margin = 0.5;
   cfg.swap = true;  // Enable distance swap
   cfg.mean = true;

**PyTorch Member**: ``torch::nn::TripletMarginLossImpl* m_triplet_margin``

20. TripletMarginWithDistanceLoss
----------------------------------------------------------------------------------------------------

**Formula**: L = max(0, distance(a, p) - distance(a, n) + margin)

**Use Cases**: Metric learning with custom distance functions

**Parameters**:
- ``margin``: Margin value (default: 1.0)
- ``swap``: Use distance swap (default: false)
- ``reduction``: "mean", "sum", or "none"

**PyTorch Member**: ``torch::nn::TripletMarginWithDistanceLossImpl* m_triplet_margin_with_distance``

All 6 Optimizers
====================================================================================================

1. Adam (Adaptive Moment Estimation)
----------------------------------------------------------------------------------------------------

**Algorithm**: Combines momentum and RMSProp adaptive learning rates

**Parameters** (via ``optimizer_params_t``):
- ``learning_rate``: Learning rate (default: 0.001)
- ``beta1``: Exponential decay rate for 1st moment (default: 0.9)
- ``beta2``: Exponential decay rate for 2nd moment (default: 0.999)
- ``epsilon``: Numerical stability term (default: 1e-8)
- ``weight_decay``: L2 regularization (default: 0)

**Use Cases**: Most popular optimizer, good default choice

**Builder Method**: ``void build_adam(optimizer_params_t* op, std::vector<torch::Tensor>* params)``

**PyTorch Member**: ``torch::optim::Adam* m_adam``

2. Adagrad
----------------------------------------------------------------------------------------------------

**Algorithm**: Adaptive learning rate per parameter

**Parameters**:
- ``learning_rate``: Learning rate (default: 0.01)
- ``epsilon``: Numerical stability (default: 1e-10)
- ``weight_decay``: L2 regularization (default: 0)

**Use Cases**: Sparse data, NLP tasks

**Builder Method**: ``void build_adagrad(optimizer_params_t* op, std::vector<torch::Tensor>* params)``

**PyTorch Member**: ``torch::optim::Adagrad* m_adagrad``

3. AdamW
----------------------------------------------------------------------------------------------------

**Algorithm**: Adam with decoupled weight decay

**Parameters**: Same as Adam

**Use Cases**: Better weight decay than Adam, transformer models

**Builder Method**: ``void build_adamw(optimizer_params_t* op, std::vector<torch::Tensor>* params)``

**PyTorch Member**: ``torch::optim::AdamW* m_adamw``

4. LBFGS (Limited-memory BFGS)
----------------------------------------------------------------------------------------------------

**Algorithm**: Quasi-Newton method

**Parameters**:
- ``learning_rate``: Learning rate (default: 1.0)
- ``max_iter``: Maximum iterations per optimization step (default: 20)
- ``max_eval``: Maximum function evaluations (default: 25)
- ``tolerance_grad``: Termination tolerance on gradient (default: 1e-7)
- ``tolerance_change``: Termination tolerance on function value (default: 1e-9)
- ``history_size``: Update history size (default: 100)

**Use Cases**: Small datasets, full-batch optimization

**Builder Method**: ``void build_lbfgs(optimizer_params_t* op, std::vector<torch::Tensor>* params)``

**PyTorch Member**: ``torch::optim::LBFGS* m_lbfgs``

5. RMSprop
----------------------------------------------------------------------------------------------------

**Algorithm**: Root Mean Square Propagation

**Parameters**:
- ``learning_rate``: Learning rate (default: 0.01)
- ``alpha``: Smoothing constant (default: 0.99)
- ``epsilon``: Numerical stability (default: 1e-8)
- ``weight_decay``: L2 regularization (default: 0)
- ``momentum``: Momentum factor (default: 0)

**Use Cases**: RNNs, non-stationary objectives

**Builder Method**: ``void build_rmsprop(optimizer_params_t* op, std::vector<torch::Tensor>* params)``

**PyTorch Member**: ``torch::optim::RMSprop* m_rmsprop``

6. SGD (Stochastic Gradient Descent)
----------------------------------------------------------------------------------------------------

**Algorithm**: Classic gradient descent with momentum

**Parameters**:
- ``learning_rate``: Learning rate (required)
- ``momentum``: Momentum factor (default: 0)
- ``dampening``: Dampening for momentum (default: 0)
- ``weight_decay``: L2 regularization (default: 0)
- ``nesterov``: Enable Nesterov momentum (default: false)

**Use Cases**: Simple problems, baseline comparisons

**Builder Method**: ``void build_sgd(optimizer_params_t* op, std::vector<torch::Tensor>* params)``

**PyTorch Member**: ``torch::optim::SGD* m_sgd``

Learning Rate Schedulers
====================================================================================================

1. StepLR
----------------------------------------------------------------------------------------------------

**Algorithm**: Decay learning rate by gamma every step_size epochs

**Parameters**:
- ``step_size``: Period of learning rate decay
- ``gamma``: Multiplicative factor of learning rate decay (default: 0.1)

**PyTorch Member**: ``torch::optim::StepLR* m_steplr``

2. ReduceLROnPlateau
----------------------------------------------------------------------------------------------------

**Algorithm**: Reduce learning rate when metric plateaus

**Parameters**:
- ``mode``: "min" or "max"
- ``factor``: Factor by which to reduce learning rate (default: 0.1)
- ``patience``: Number of epochs with no improvement (default: 10)
- ``threshold``: Threshold for measuring improvement (default: 1e-4)

**PyTorch Member**: ``torch::optim::ReduceLROnPlateauScheduler* m_rlp``

Usage Examples
====================================================================================================

Basic Loss Function Usage
----------------------------------------------------------------------------------------------------

.. code-block:: cpp

   #include <templates/lossfx.h>
   
   // Create loss function
   lossfx loss_fn;
   
   // Configure BCE loss
   loss_fn.lss_cfg.mean = true;
   loss_fn.build_loss_function(loss_enum::bce);
   
   // Compute loss
   torch::Tensor predictions = model->forward(inputs);
   torch::Tensor loss_value = loss_fn.loss(&predictions, &targets);

Multi-Class Classification with CrossEntropy
----------------------------------------------------------------------------------------------------

.. code-block:: cpp

   lossfx loss_fn;
   
   // Configure CrossEntropy with label smoothing
   loss_fn.lss_cfg.mean = true;
   loss_fn.lss_cfg.smoothing = 0.1;  // 10% label smoothing
   loss_fn.lss_cfg.ignore = 0;  // Ignore padding token (index 0)
   loss_fn.build_loss_function(loss_enum::cross_entropy);
   
   torch::Tensor logits = model->forward(inputs);
   torch::Tensor loss = loss_fn.loss(&logits, &targets);

Optimizer with Scheduler
----------------------------------------------------------------------------------------------------

.. code-block:: cpp

   // Setup optimizer configuration
   optimizer_params_t opt_config;
   opt_config.learning_rate = 0.001;
   opt_config.beta1 = 0.9;
   opt_config.beta2 = 0.999;
   opt_config.weight_decay = 1e-5;
   
   // Build Adam optimizer
   lossfx loss_fn;
   std::vector<torch::Tensor> parameters = model->parameters();
   torch::optim::Optimizer* optimizer = loss_fn.build_optimizer(&opt_config, &parameters);
   
   // Add learning rate scheduler
   loss_fn.build_scheduler(&opt_config, optimizer);
   
   // Training loop
   for (int epoch = 0; epoch < num_epochs; ++epoch) {
       torch::Tensor loss = loss_fn.loss(&predictions, &targets);
       optimizer->zero_grad();
       loss.backward();
       optimizer->step();
       loss_fn.step();  // Step scheduler
   }

Private Implementation Details
====================================================================================================

Loss Function Configuration Methods
----------------------------------------------------------------------------------------------------

The ``lossfx`` class uses template methods to configure loss functions based on their parameters:

.. code-block:: cpp

   template <typename g>
   void _dress_reduction(g* imx, loss_opt* params);  // Set reduction mode
   
   template <typename g>
   void _dress_ignore(g* imx, loss_opt* params);  // Set ignore index
   
   template <typename g>
   void _dress_smoothing(g* imx, loss_opt* params);  // Set label smoothing
   
   template <typename g>
   void _dress_margin(g* imx, loss_opt* params);  // Set margin
   
   template <typename g>
   void _dress_delta(g* imx, loss_opt* params);  // Set delta (Huber loss)

These are called automatically by the ``build_fx_loss()`` methods for each loss type.


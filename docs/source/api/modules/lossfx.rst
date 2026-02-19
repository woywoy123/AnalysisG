Loss Functions Module (C++)
============================

The Loss Functions module provides training optimization and loss function support.

Overview
--------

Located in ``src/AnalysisG/modules/lossfx/``, this module implements C++ loss functions 
and optimizer configuration:

- OptimizerConfig for training configuration
- Integration with LibTorch optimizers
- Learning rate scheduling
- Loss function definitions

Purpose
-------

The lossfx module enables:

- Configuring training hyperparameters
- Selecting optimization algorithms
- Defining learning rate schedules
- Computing loss functions

Implementation Files
--------------------

**C++ Implementation**

- ``src/AnalysisG/modules/lossfx/cxx/*.cxx`` - Loss function implementations
- ``src/AnalysisG/modules/lossfx/include/templates/*.h`` - OptimizerConfig headers

**Python Binding**

- ``src/AnalysisG/core/lossfx.pyx`` - Cython wrapper
- ``src/AnalysisG/core/lossfx.pxd`` - Cython declarations

Key Classes
-----------

**OptimizerConfig**

Configuration class for training:

.. code-block:: cpp

   class optimizer_config {
   public:
       // Optimizer selection
       std::string optimizer;  // "Adam", "SGD", "RMSprop", etc.
       
       // Learning rate
       double lr;
       double lr_decay;
       
       // Scheduler
       std::string scheduler;  // "StepLR", "ExponentialLR", etc.
       int step_size;
       double gamma;
       
       // Optimizer parameters
       double weight_decay;
       double momentum;
       double eps;
       double alpha;
       bool amsgrad;
       
       // Advanced settings
       bool nesterov;
       double dampening;
       bool centered;
       
       // LBFGS parameters
       int max_iter;
       int max_eval;
       double tolerance_grad;
       double tolerance_change;
       int history_size;
   };

Supported Optimizers
--------------------

**Adam**

Adaptive Moment Estimation:

.. code-block:: cpp

   config.optimizer = "Adam";
   config.lr = 0.001;
   config.weight_decay = 1e-5;
   config.eps = 1e-8;
   config.amsgrad = false;

**SGD**

Stochastic Gradient Descent:

.. code-block:: cpp

   config.optimizer = "SGD";
   config.lr = 0.01;
   config.momentum = 0.9;
   config.weight_decay = 1e-4;
   config.nesterov = true;

**RMSprop**

Root Mean Square Propagation:

.. code-block:: cpp

   config.optimizer = "RMSprop";
   config.lr = 0.01;
   config.alpha = 0.99;
   config.eps = 1e-8;
   config.weight_decay = 0;
   config.momentum = 0;
   config.centered = false;

**AdamW**

Adam with Weight Decay:

.. code-block:: cpp

   config.optimizer = "AdamW";
   config.lr = 0.001;
   config.weight_decay = 0.01;

**Adagrad**

Adaptive Gradient:

.. code-block:: cpp

   config.optimizer = "Adagrad";
   config.lr = 0.01;
   config.lr_decay = 0;
   config.weight_decay = 0;

**LBFGS**

Limited-memory BFGS:

.. code-block:: cpp

   config.optimizer = "LBFGS";
   config.lr = 1.0;
   config.max_iter = 20;
   config.max_eval = 25;
   config.tolerance_grad = 1e-5;
   config.tolerance_change = 1e-9;
   config.history_size = 100;

Learning Rate Schedulers
-------------------------

**StepLR**

Decay learning rate by gamma every step_size epochs:

.. code-block:: cpp

   config.scheduler = "StepLR";
   config.step_size = 30;
   config.gamma = 0.1;

**ExponentialLR**

Exponential decay:

.. code-block:: cpp

   config.scheduler = "ExponentialLR";
   config.gamma = 0.95;

**ReduceLROnPlateau**

Reduce when metric plateaus:

.. code-block:: cpp

   config.scheduler = "ReduceLROnPlateau";
   config.patience = 10;
   config.factor = 0.1;

Usage Example
-------------

.. code-block:: cpp

   #include <templates/optimizer_config.h>
   
   // Create config
   optimizer_config config;
   
   // Set optimizer
   config.optimizer = "Adam";
   config.lr = 0.001;
   config.weight_decay = 1e-5;
   
   // Set scheduler
   config.scheduler = "StepLR";
   config.step_size = 10;
   config.gamma = 0.1;
   
   // Use in training
   auto model = create_model();
   auto optimizer = create_optimizer(model.parameters(), config);
   auto scheduler = create_scheduler(optimizer, config);

Integration with Python
-----------------------

The C++ OptimizerConfig is wrapped in Python:

.. code-block:: python

   from AnalysisG.core.lossfx import OptimizerConfig
   
   config = OptimizerConfig()
   config.Optimizer = "Adam"
   config.lr = 0.001
   config.Scheduler = "StepLR"
   config.step_size = 10
   config.gamma = 0.1

Loss Functions
--------------

Common loss functions supported:

**Classification**

- Cross-entropy loss
- Binary cross-entropy
- Focal loss
- Label smoothing

**Regression**

- Mean squared error (MSE)
- Mean absolute error (MAE)
- Huber loss
- Smooth L1 loss

**Custom Loss**

Define custom loss functions:

.. code-block:: cpp

   class custom_loss {
   public:
       torch::Tensor forward(
           torch::Tensor predictions,
           torch::Tensor targets
       ) {
           // Compute loss
           return loss;
       }
   };

Gradient Clipping
-----------------

Control gradient magnitudes:

.. code-block:: cpp

   config.gradient_clip_value = 1.0;  // Clip by value
   config.gradient_clip_norm = 1.0;   // Clip by norm

See Also
--------

* :doc:`optimizer` - Optimizer module
* :doc:`model` - Model template
* :doc:`metric` - Metric template

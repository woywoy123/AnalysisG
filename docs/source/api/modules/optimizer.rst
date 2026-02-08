Optimizer Module
================

The optimizer module provides training optimization algorithms for machine learning models.

Overview
--------

Located in ``src/AnalysisG/modules/optimizer/``, this module implements:

- Optimizer configuration
- Learning rate scheduling
- Gradient clipping
- Weight decay
- Momentum optimization
- Adam and other optimizers

Integration with Analysis
--------------------------

.. code-block:: python

   from AnalysisG.core import Analysis, ModelTemplate, OptimizerConfig
   
   # Configure optimizer
   config = OptimizerConfig()
   config.learning_rate = 0.001
   config.weight_decay = 1e-5
   config.momentum = 0.9
   
   # Add to analysis
   ana = Analysis()
   ana.AddModel(my_model, config, "training_run")

OptimizerConfig Class
---------------------

.. class:: OptimizerConfig

   Configuration class for optimization parameters.

   **Properties**

   .. attribute:: learning_rate
      :type: float

      Learning rate for optimizer. Default: 0.001

   .. attribute:: weight_decay
      :type: float

      L2 regularization parameter. Default: 0.0

   .. attribute:: momentum
      :type: float

      Momentum factor for SGD-based optimizers. Default: 0.9

   .. attribute:: beta1
      :type: float

      Beta1 parameter for Adam optimizer. Default: 0.9

   .. attribute:: beta2
      :type: float

      Beta2 parameter for Adam optimizer. Default: 0.999

   .. attribute:: epsilon
      :type: float

      Epsilon for numerical stability. Default: 1e-8

   .. attribute:: optimizer_type
      :type: str

      Type of optimizer: "adam", "sgd", "adamw". Default: "adam"

   .. attribute:: scheduler_type
      :type: str

      Learning rate scheduler: "step", "cosine", "plateau". Default: None

   .. attribute:: scheduler_params
      :type: dict

      Parameters for the scheduler.

Learning Rate Scheduling
-------------------------

Step Scheduler
~~~~~~~~~~~~~~

.. code-block:: python

   config = OptimizerConfig()
   config.scheduler_type = "step"
   config.scheduler_params = {
       'step_size': 30,  # Epochs between lr drops
       'gamma': 0.1      # Multiplication factor
   }

Cosine Annealing
~~~~~~~~~~~~~~~~

.. code-block:: python

   config = OptimizerConfig()
   config.scheduler_type = "cosine"
   config.scheduler_params = {
       'T_max': 100,     # Maximum iterations
       'eta_min': 1e-6   # Minimum learning rate
   }

Plateau Scheduler
~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = OptimizerConfig()
   config.scheduler_type = "plateau"
   config.scheduler_params = {
       'mode': 'min',    # Minimize metric
       'factor': 0.5,    # LR reduction factor
       'patience': 10    # Epochs to wait
   }

Gradient Clipping
-----------------

.. code-block:: python

   config = OptimizerConfig()
   config.gradient_clip_value = 1.0  # Clip gradients to [-1, 1]
   # or
   config.gradient_clip_norm = 1.0   # Clip gradient norm to 1.0

C++ Implementation
------------------

The optimizer module has a C++ backend (``optimizer.cxx``) that:

- Integrates with LibTorch optimizers
- Provides efficient parameter updates
- Handles distributed training
- Manages optimizer state

See Also
--------

* :doc:`../core/analysis` - Analysis class
* :doc:`../core/templates` - Model templates

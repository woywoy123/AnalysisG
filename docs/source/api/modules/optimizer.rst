Optimizer Module
================

The optimizer module provides training optimization configuration for machine learning models.

Overview
--------

OptimizerConfig is located in ``src/AnalysisG/core/lossfx.pyx`` and provides configuration
for LibTorch optimizers including:

- Learning rate and decay
- Optimizer selection (SGD, Adam, RMSprop, etc.)
- Scheduler selection (StepLR, etc.)
- Momentum and weight decay
- Advanced optimizer parameters

Integration with Analysis
--------------------------

.. code-block:: python

   from AnalysisG import Analysis, OptimizerConfig
   from AnalysisG.models import Grift
   
   # Configure optimizer
   config = OptimizerConfig()
   config.Optimizer = "Adam"
   config.Scheduler = "StepLR"
   config.lr = 0.001
   config.weight_decay = 1e-5
   config.momentum = 0.9
   config.step_size = 10
   config.gamma = 0.1
   
   # Add to analysis
   model = Grift()
   ana = Analysis()
   ana.AddModel(model, config, "training_run")

OptimizerConfig Class
---------------------

.. class:: OptimizerConfig

   Configuration class for optimization parameters.

   Located in ``AnalysisG.core.lossfx``.

   **Properties**

   **Optimizer and Scheduler**

   .. attribute:: Optimizer
      :type: str

      Type of optimizer. Examples: "Adam", "SGD", "RMSprop", "Adagrad", "LBFGS"

   .. attribute:: Scheduler
      :type: str

      Type of learning rate scheduler. Examples: "StepLR", "ExponentialLR"

   **Learning Rate**

   .. attribute:: lr
      :type: float

      Learning rate for the optimizer.

   .. attribute:: lr_decay
      :type: float

      Learning rate decay factor.

   .. attribute:: step_size
      :type: int

      Number of epochs between learning rate updates (for StepLR).

   .. attribute:: gamma
      :type: float

      Multiplicative factor for learning rate decay.

   **Regularization**

   .. attribute:: weight_decay
      :type: float

      L2 regularization parameter (weight decay).

   **Momentum Parameters**

   .. attribute:: momentum
      :type: float

      Momentum factor for SGD and related optimizers.

   .. attribute:: dampening
      :type: float

      Dampening for momentum.

   .. attribute:: nesterov
      :type: bool

      Whether to use Nesterov momentum.

   **Adam/RMSprop Parameters**

   .. attribute:: alpha
      :type: float

      Smoothing constant for RMSprop.

   .. attribute:: eps
      :type: float

      Epsilon for numerical stability.

   .. attribute:: amsgrad
      :type: bool

      Whether to use AMSGrad variant of Adam.

   .. attribute:: centered
      :type: bool

      Whether to use centered version of RMSprop.

   **Adagrad Parameters**

   .. attribute:: initial_accumulator_value
      :type: float

      Initial value of the accumulator for Adagrad.

   **LBFGS Parameters**

   .. attribute:: max_iter
      :type: int

      Maximum number of iterations per optimization step.

   .. attribute:: max_eval
      :type: int

      Maximum number of function evaluations per optimization step.

   .. attribute:: tolerance_grad
      :type: float

      Termination tolerance on first order optimality.

   .. attribute:: tolerance_change
      :type: float

      Termination tolerance on function value/parameter changes.

   .. attribute:: history_size
      :type: int

      Update history size for LBFGS.

Example Configurations
----------------------

Adam Optimizer
~~~~~~~~~~~~~~

.. code-block:: python

   config = OptimizerConfig()
   config.Optimizer = "Adam"
   config.lr = 0.001
   config.weight_decay = 1e-5
   config.eps = 1e-8
   config.amsgrad = False

SGD with Momentum
~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = OptimizerConfig()
   config.Optimizer = "SGD"
   config.lr = 0.01
   config.momentum = 0.9
   config.weight_decay = 5e-4
   config.nesterov = True

With Learning Rate Scheduling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config = OptimizerConfig()
   config.Optimizer = "Adam"
   config.Scheduler = "StepLR"
   config.lr = 0.001
   config.step_size = 10  # Decay every 10 epochs
   config.gamma = 0.1     # Multiply lr by 0.1

RMSprop
~~~~~~~

.. code-block:: python

   config = OptimizerConfig()
   config.Optimizer = "RMSprop"
   config.lr = 0.001
   config.alpha = 0.99
   config.eps = 1e-8
   config.weight_decay = 0
   config.momentum = 0
   config.centered = False

See Also
--------

* :doc:`../core/analysis` - Analysis class
* :doc:`../core/templates` - Model templates

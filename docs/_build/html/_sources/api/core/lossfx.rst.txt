OptimizerConfig (Python)
========================

The ``OptimizerConfig`` Cython class wraps the C++ ``optimizer_params_t``
struct and is used to configure the optimiser and learning-rate scheduler
when calling ``Analysis.AddModel``.

.. code-block:: python

   from AnalysisG.core.lossfx import OptimizerConfig

   op = OptimizerConfig()
   op.Optimizer = "Adam"
   op.Scheduler = "StepLR"
   op.lr = 1e-3
   op.step_size = 10
   op.gamma = 0.5

Optimiser / Scheduler Selection
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Property
     - Default
     - Description
   * - ``Optimizer``
     - ``''``
     - Optimiser name: ``"Adam"``, ``"SGD"``, ``"RMSprop"``,
       ``"AdaGrad"``, ``"LBFGS"``.
   * - ``Scheduler``
     - ``''``
     - LR-scheduler name: ``"StepLR"``, ``"CyclicLR"``,
       ``"ExponentialLR"``.

Hyper-parameter Properties
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Property
     - Default
     - Description
   * - ``lr``
     - ``0.0``
     - Learning rate.
   * - ``lr_decay``
     - ``0.0``
     - L2 learning-rate decay (AdaGrad).
   * - ``weight_decay``
     - ``0.0``
     - L2 regularisation weight decay.
   * - ``eps``
     - ``0.0``
     - Numerical stability epsilon (Adam, RMSprop).
   * - ``alpha``
     - ``0.0``
     - RMSprop smoothing constant.
   * - ``momentum``
     - ``0.0``
     - SGD/RMSprop momentum factor.
   * - ``dampening``
     - ``0.0``
     - SGD dampening for momentum.
   * - ``betas``
     - ``(0, 0)``
     - Adam ``(β₁, β₂)`` tuple.
   * - ``amsgrad``
     - ``False``
     - Use AMSGrad variant of Adam.
   * - ``nesterov``
     - ``False``
     - Use Nesterov momentum (SGD).
   * - ``centered``
     - ``False``
     - Use centred RMSprop.
   * - ``initial_accumulator_value``
     - ``0.0``
     - AdaGrad initial accumulator value.
   * - ``step_size``
     - ``1``
     - StepLR period in epochs.
   * - ``gamma``
     - ``0.1``
     - StepLR/ExponentialLR multiplicative decay factor.
   * - ``max_iter``
     - ``0``
     - LBFGS maximum iterations per step.
   * - ``max_eval``
     - ``0``
     - LBFGS maximum function evaluations per step.
   * - ``history_size``
     - ``0``
     - LBFGS history size.
   * - ``tolerance_grad``
     - ``0.0``
     - LBFGS gradient-norm convergence tolerance.
   * - ``tolerance_change``
     - ``0.0``
     - LBFGS step-size convergence tolerance.

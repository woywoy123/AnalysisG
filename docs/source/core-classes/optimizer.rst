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

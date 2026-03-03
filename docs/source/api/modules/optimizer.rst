Optimizer Module
================

The ``optimizer`` class orchestrates the PyTorch training loop.  It
receives a populated ``dataloader``, creates per-k-fold model copies,
runs training / validation / evaluation epochs, and emits
``model_report`` structs for the metrics layer.

It inherits from ``tools`` and ``notification`` to provide file-system
utilities and coloured terminal logging.

Key methods:

- ``import_dataloader(dl)`` — bind the data source
- ``import_model_sessions(models)`` — register model + ``optimizer_params_t`` pairs
- ``training_loop(k, epoch)`` — run one training epoch for fold ``k``
- ``validation_loop(k, epoch)`` — run one validation epoch
- ``evaluation_loop(k, epoch)`` — run one evaluation (test) epoch
- ``launch_model(k)`` — build the k-fold model copy and PyTorch optimizer

.. doxygenclass:: optimizer
   :project: AnalysisG
   :members:
   :protected-members:
   :undoc-members:

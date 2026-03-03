Optimizer Module
================

The ``optimizer`` class orchestrates the complete training loop for one model
across all k-folds.  It delegates data loading to ``dataloader``, metric
capture to ``metrics``, and loss computation to the per-feature ``lossfx``
objects embedded in ``model_template``.

Class: ``optimizer``
---------------------

**Header:** ``<generators/optimizer.h>``

**Inheritance:** ``tools``, ``notification``

Public Fields
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Field
     - Type
     - Description
   * - ``m_settings``
     - ``settings_t``
     - Framework training settings (epochs, kfolds, batch-size, threads, …).
   * - ``kfold_sessions``
     - ``std::map<int, model_template*>``
     - Maps fold index → cloned model instance.
   * - ``reports``
     - ``std::map<std::string, model_report*>``
     - Maps model-name → training report struct.
   * - ``metric``
     - ``metrics*``
     - Pointer to the metrics engine (not owned).
   * - ``loader``
     - ``dataloader*``
     - Pointer to the data loader (not owned).

Public Methods
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Signature
     - Description
   * - ``void import_dataloader(dataloader* dl)``
     - Sets the ``loader`` pointer and links ``m_settings``.
   * - ``void import_model_sessions(std::tuple<model_template*, optimizer_params_t*>* models)``
     - Clones the model for each k-fold and initialises optimisers and
       schedulers from *models*.
   * - ``void training_loop(int k, int epoch)``
     - Runs one full training epoch for fold *k*.
   * - ``void validation_loop(int k, int epoch)``
     - Runs one full validation epoch for fold *k*.
   * - ``void evaluation_loop(int k, int epoch)``
     - Runs one full evaluation (test-set) pass for fold *k*.
   * - ``void launch_model(int k)``
     - Executes all epochs (train + validate + evaluate) for fold *k*.

Metrics Module
==============

The ``metrics`` class is the framework's training-analytics engine.  It
registers one ``analytics_t`` object per model/kfold pair and captures
loss, accuracy, and invariant-mass histograms during each epoch, then dumps
ROOT/PDF plots to the output directory.

Struct: ``analytics_t``
------------------------

**Header:** ``<metrics/metrics.h>``

``analytics_t`` holds all ROOT histograms for one model/kfold pair.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Field
     - Type
     - Description
   * - ``model``
     - ``model_template*``
     - Pointer to the associated model (not owned).
   * - ``report``
     - ``model_report*``
     - Pointer to the model-report struct (owned; freed by ``purge()``).
   * - ``this_epoch``
     - ``int``
     - Current epoch counter for this model/kfold.
   * - ``loss_graph``
     - ``std::map<mode_enum, std::map<std::string, TH1F*>>``
     - Per-mode (train/valid/eval) loss histograms for graph-level features.
   * - ``loss_node``
     - same type
     - Node-level loss histograms.
   * - ``loss_edge``
     - same type
     - Edge-level loss histograms.
   * - ``accuracy_graph``
     - same type
     - Graph-level accuracy histograms.
   * - ``accuracy_node``
     - same type
     - Node-level accuracy histograms.
   * - ``accuracy_edge``
     - same type
     - Edge-level accuracy histograms.
   * - ``pred_mass_edge``
     - same type
     - Predicted invariant-mass histograms for edge features.
   * - ``truth_mass_edge``
     - same type
     - Truth invariant-mass histograms for edge features.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - ``void purge()``
     - Deletes all histogram pointers and the ``report`` pointer.
   * - ``void destroy(std::map<mode_enum, std::map<std::string, TH1F*>>* data)``
     - Iterates and deletes all ``TH1F`` pointers in *data*.

Class: ``metrics``
------------------

**Header:** ``<metrics/metrics.h>``

**Inheritance:** ``tools``, ``notification``

Public Fields
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Field
     - Type
     - Description
   * - ``output_path``
     - ``std::string``
     - Directory to which plots are written.
   * - ``m_settings``
     - ``settings_t``
     - Framework settings struct (kfolds, batch-size, etc.).
   * - ``colors_h``
     - ``std::vector<Color_t>``
     - ROOT colour list for multi-kfold overlays (red, green, blue, cyan, violet, orange, coffee, aurora).

Public Methods
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Signature
     - Description
   * - ``model_report* register_model(model_template* model, int kfold)``
     - Creates and stores an ``analytics_t`` entry for *model* at fold *kfold*.
       Returns the freshly allocated ``model_report``.
   * - ``void capture(mode_enum mode, int kfold, int epoch, int smpl_len)``
     - Captures loss, accuracy, and mass histograms from the current model
       outputs for *mode* (train/valid/eval) at the given *kfold*/*epoch*.
       *smpl_len* is the number of events in the batch (used for normalisation).
   * - ``void dump_plots(int k)``
     - Writes all loss + accuracy + mass plots for fold *k* to ``output_path``.
   * - ``void dump_loss_plots(int k)``
     - Writes only loss plots for fold *k*.
   * - ``void dump_accuracy_plots(int k)``
     - Writes only accuracy plots for fold *k*.
   * - ``void dump_mass_plots(int k)``
     - Writes only invariant-mass plots for fold *k*.

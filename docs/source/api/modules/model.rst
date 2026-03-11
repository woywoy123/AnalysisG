Model Template
==============

``model_template`` is the base class for all user-defined GNN models.  It
inherits from both ``notification`` (for logging) and ``tools`` (for utilities).
Subclasses must override ``forward(graph_t* data)`` and call
``register_module`` to add PyTorch ``Sequential`` networks.

Class: ``model_template``
--------------------------

**Header:** ``<templates/model_template.h>``

**Inheritance:** ``notification``, ``tools``

Output Feature Maps (``cproperty``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These properties map feature-name → loss-function-name and are set from the
Python ``OptimizerConfig`` via the Cython wrapper.

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Property
     - Value Type
     - Description
   * - ``o_graph``
     - ``std::map<std::string, std::string>``
     - Map of graph-level output feature names to loss-function names.
   * - ``o_node``
     - ``std::map<std::string, std::string>``
     - Map of node-level output feature names to loss-function names.
   * - ``o_edge``
     - ``std::map<std::string, std::string>``
     - Map of edge-level output feature names to loss-function names.

Input Feature Lists (``cproperty``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Property
     - Value Type
     - Description
   * - ``i_graph``
     - ``std::vector<std::string>``
     - Names of graph-level data features to fetch from ``graph_t``.
   * - ``i_node``
     - ``std::vector<std::string>``
     - Names of node-level data features to fetch from ``graph_t``.
   * - ``i_edge``
     - ``std::vector<std::string>``
     - Names of edge-level data features to fetch from ``graph_t``.

Device / Identity Properties (``cproperty``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Property
     - Type
     - Description
   * - ``device``
     - ``std::string``
     - Device string, e.g. ``"cpu"`` or ``"cuda:0"``.
   * - ``device_index``
     - ``int``
     - Numeric CUDA device index (``-1`` = CPU).
   * - ``name``
     - ``std::string``
     - Model name, used as the HDF5 weight-file subdirectory.

Public Fields
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Field
     - Type
     - Description
   * - ``kfold``
     - ``int``
     - Current k-fold iteration index (set by ``optimizer``).
   * - ``epoch``
     - ``int``
     - Current epoch (set by ``optimizer``).
   * - ``is_mc``
     - ``bool``
     - Whether the current batch contains Monte Carlo data.  Default ``false``.
   * - ``use_pkl``
     - ``bool``
     - Use Python-pickle checkpointing instead of HDF5.  Default ``false``.
   * - ``inference_mode``
     - ``bool``
     - Set to ``true`` during inference; disables gradient tracking.  Default ``false``.
   * - ``enable_anomaly``
     - ``bool``
     - Enable PyTorch anomaly detection.  Default ``false``.
   * - ``retain_graph``
     - ``bool``
     - Retain computation graph for multiple backwards passes.  Default ``false``.
   * - ``model_checkpoint_path``
     - ``std::string``
     - Directory for weight checkpoints.  Default ``""``.
   * - ``weight_name``
     - ``std::string``
     - Name of the HDF5 event-weight field.  Default ``"event_weight"``.
   * - ``tree_name``
     - ``std::string``
     - ROOT tree name used for weight look-up.  Default ``"nominal"``.

Virtual Methods (Override in Subclass)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Signature
     - Description
   * - ``virtual void forward(graph_t* data)``
     - **Primary override.** Implement the GNN forward pass here.
       Call ``data->get_data_node("name", this)`` etc. to retrieve input
       tensors, then call ``prediction_node_feature(...)`` to store outputs.
   * - ``virtual model_template* clone()``
     - Override to return a heap-allocated copy of the model.
   * - ``virtual void train_sequence(bool mode)``
     - Override to switch sub-modules between train/eval modes.

Framework Methods
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Signature
     - Description
   * - ``void register_module(torch::nn::Sequential* data)``
     - Registers a ``Sequential`` network; parameters are included in the
       optimizer parameter group.
   * - ``void register_module(torch::nn::Sequential* data, mlp_init weight_init)``
     - Registers a ``Sequential`` network and initialises weights using the
       given ``mlp_init`` strategy (Xavier, Kaiming, Normal, …).
   * - ``void prediction_graph_feature(std::string name, torch::Tensor t)``
     - Stores the graph-level prediction tensor *t* under key *name*.
   * - ``void prediction_node_feature(std::string name, torch::Tensor t)``
     - Stores the node-level prediction tensor *t* under key *name*.
   * - ``void prediction_edge_feature(std::string name, torch::Tensor t)``
     - Stores the edge-level prediction tensor *t* under key *name*.
   * - ``void prediction_extra(std::string name, torch::Tensor t)``
     - Stores an auxiliary output tensor (not matched to a truth feature).
   * - ``torch::Tensor* compute_loss(std::string name, graph_enum type)``
     - Computes and returns the loss for output feature *name* of *type*
       (``graph_enum::data_graph``, ``data_node``, or ``data_edge``).
   * - ``void evaluation_mode(bool mode = true)``
     - Switches all registered modules to eval/train mode.
   * - ``void save_state()``
     - Saves the current model state (weights) to the checkpoint directory.
   * - ``bool restore_state()``
     - Restores the model state from the checkpoint directory.  Returns
       ``false`` if no checkpoint exists.
   * - ``void check_features(graph_t*)``
     - Verifies that all requested input features exist in the ``graph_t``
       object and prints warnings for any that are missing.
   * - ``void set_optimizer(std::string name)``
     - Sets the optimizer type by name (``"Adam"``, ``"SGD"``, etc.).
   * - ``void initialize(optimizer_params_t*)``
     - Builds the PyTorch optimizer using the supplied parameters.

Example ``forward`` Implementation::

    void MyModel::forward(graph_t* data) {
        // Get input tensors
        torch::Tensor* pt   = data->get_data_node("pt", this);
        torch::Tensor* eta  = data->get_data_node("eta", this);
        torch::Tensor* ei   = data->get_edge_index(this);

        // Run GNN layers
        torch::Tensor x = torch::cat({*pt, *eta}, 1);
        x = this->gnn_layer->forward(x);

        // Store output predictions
        this->prediction_node_feature("node_cls", x);
    }

Loss Functions Module
=====================

``lossfx`` wraps PyTorch's loss functions, optimisers, and learning-rate
schedulers into a single C++ class, and provides weight-initialisation
utilities for ``torch::nn::Sequential`` networks.  It is used internally by
``model_template`` and exposed to Python via ``OptimizerConfig`` in
``AnalysisG.core.lossfx``.

Class: ``lossfx``
-----------------

**Header:** ``<templates/lossfx.h>``

**Inheritance:** ``tools``, ``notification``

Supported Loss Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

The following PyTorch loss functions are selectable by name (case-insensitive):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Name string
     - PyTorch class
   * - ``"bce"``
     - ``torch::nn::BCELoss``
   * - ``"bce_with_logits"``
     - ``torch::nn::BCEWithLogitsLoss``
   * - ``"cosine_embedding"``
     - ``torch::nn::CosineEmbeddingLoss``
   * - ``"cross_entropy"``
     - ``torch::nn::CrossEntropyLoss``
   * - ``"ctc"``
     - ``torch::nn::CTCLoss``
   * - ``"hinge_embedding"``
     - ``torch::nn::HingeEmbeddingLoss``
   * - ``"huber"``
     - ``torch::nn::HuberLoss``
   * - ``"kl_div"``
     - ``torch::nn::KLDivLoss``
   * - ``"l1"``
     - ``torch::nn::L1Loss``
   * - ``"margin_ranking"``
     - ``torch::nn::MarginRankingLoss``
   * - ``"mse"``
     - ``torch::nn::MSELoss``
   * - ``"multi_label_margin"``
     - ``torch::nn::MultiLabelMarginLoss``
   * - ``"multi_label_soft_margin"``
     - ``torch::nn::MultiLabelSoftMarginLoss``
   * - ``"multi_margin"``
     - ``torch::nn::MultiMarginLoss``
   * - ``"nll"``
     - ``torch::nn::NLLLoss``
   * - ``"poisson_nll"``
     - ``torch::nn::PoissonNLLLoss``
   * - ``"smooth_l1"``
     - ``torch::nn::SmoothL1Loss``
   * - ``"soft_margin"``
     - ``torch::nn::SoftMarginLoss``
   * - ``"triplet_margin"``
     - ``torch::nn::TripletMarginLoss``
   * - ``"triplet_margin_with_distance"``
     - ``torch::nn::TripletMarginWithDistanceLoss``

Supported Optimisers
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name string (``opt_enum``)
     - PyTorch class
   * - ``"Adam"``
     - ``torch::optim::Adam``
   * - ``"Adagrad"``
     - ``torch::optim::Adagrad``
   * - ``"AdamW"``
     - ``torch::optim::AdamW``
   * - ``"LBFGS"``
     - ``torch::optim::LBFGS``
   * - ``"RMSprop"``
     - ``torch::optim::RMSprop``
   * - ``"SGD"``
     - ``torch::optim::SGD``

Supported Learning-Rate Schedulers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name string (``scheduler_enum``)
     - PyTorch class
   * - ``"StepLR"``
     - ``torch::optim::StepLR``
   * - ``"ReduceLROnPlateau"``
     - ``torch::optim::ReduceLROnPlateauScheduler``

Public Fields
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Field
     - Type
     - Description
   * - ``variable``
     - ``std::string``
     - Name of the output feature this loss function is attached to.
   * - ``lss_cfg``
     - ``loss_opt``
     - Loss-option struct (reduction type, smoothing, margin, …).

Public Methods
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Signature
     - Description
   * - ``lossfx()``
     - Default constructor.
   * - ``lossfx(std::string var, std::string enx)``
     - Constructs with feature name *var* and loss-function name *enx*.
   * - ``torch::Tensor loss(torch::Tensor* pred, torch::Tensor* truth)``
     - Computes the loss using the configured loss function.
   * - ``torch::Tensor loss(torch::Tensor* pred, torch::Tensor* truth, loss_enum lss)``
     - Computes the loss using an explicit ``loss_enum`` value.
   * - ``bool build_loss_function()``
     - Builds the loss function from ``lss_cfg``.  Returns ``false`` on failure.
   * - ``bool build_loss_function(loss_enum lss)``
     - Builds the loss function for the given enum value.
   * - ``torch::optim::Optimizer* build_optimizer(optimizer_params_t* op, std::vector<torch::Tensor>* params)``
     - Constructs and returns the PyTorch optimiser.
   * - ``void build_scheduler(optimizer_params_t* op, torch::optim::Optimizer* opx)``
     - Wraps the given optimiser with a learning-rate scheduler.
   * - ``void weight_init(torch::nn::Sequential* data, mlp_init method)``
     - Applies the weight-initialisation strategy *method* to *data*.
   * - ``void loss_opt_string(std::string name)``
     - Parses a composite option string like ``"cross_entropy|mean"`` and
       populates ``lss_cfg``.
   * - ``void to(torch::TensorOptions*)``
     - Moves the loss function to the specified device.
   * - ``void step()``
     - Calls the learning-rate scheduler step (if a scheduler is configured).

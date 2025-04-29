.. _lossfx_h:

lossfx.h
========

**File:** ``lossfx.h``

**Brief:** Defines the ``lossfx`` class, a comprehensive utility designed to streamline the management of PyTorch loss functions, optimization algorithms, and neural network weight initialization strategies within a C++ environment.

**Details:**

This header file introduces the ``lossfx`` class, which serves as a central hub for handling various components crucial for training PyTorch models using the LibTorch C++ API. It encapsulates the logic for:

*   Translating human-readable string identifiers (like "mse" or "adam") into their corresponding internal enumeration types (``loss_enum``, ``opt_enum``) for programmatic use. This facilitates configuration loading from files or user input.
*   Computing the loss value between model predictions and ground truth targets using a wide array of standard PyTorch loss functions. The class manages the instantiation and lifecycle of these loss function modules.
*   Constructing and configuring various PyTorch optimizers (e.g., Adam, SGD, RMSprop) based on specified hyperparameters. It ensures that optimizer instances are created and managed correctly, linking them to the model parameters they need to update.
*   Applying different weight initialization schemes (like Xavier/Glorot or Kaiming/He) to the layers (specifically linear layers) within a ``torch::nn::Sequential`` module. This is essential for promoting stable and efficient training convergence.
*   Managing the memory allocation and deallocation for the created loss function modules and optimizer objects, preventing resource leaks through its destructor.
*   Transferring the instantiated loss function modules to the appropriate computational device (CPU or GPU) as specified by the user, ensuring compatibility with the rest of the model and data tensors.

The class aims to provide a simplified and robust interface for common training loop operations involving loss calculation, optimization, and model initialization in C++. It leverages internal helper functions and manages object lifecycles to reduce boilerplate code in the main training logic.

.. note::
    Include directives (``<map>``, ``<vector>``, ``<string>``, ``<tools/tools.h>``, ``<torch/torch.h>``, ``<structs/enums.h>``, ``<structs/optimizer.h>``) are omitted in this documentation as per the original source comment.

API Documentation
-----------------

.. cpp:class:: lossfx : public tools

    A manager class facilitating the use of PyTorch loss functions, optimizers, and weight initialization techniques within C++ applications.

    **Details:**

    The ``lossfx`` class inherits from a base class named ``tools``, presumably gaining access to common utility functions like string manipulation (e.g., converting strings to lowercase, as used in ``loss_string`` and ``optim_string``).

    This class functions primarily as a factory and a manager. It centralizes the creation, configuration, storage, and destruction of PyTorch optimizer objects (``torch::optim::Optimizer``) and loss function modules (e.g., ``torch::nn::MSELossImpl``). This abstraction simplifies the setup and execution of the training process for neural networks built with LibTorch.

    Key responsibilities include:

    *   Mapping string names to internal enums for selecting loss functions and optimizers.
    *   Providing a unified interface (``loss`` method) to calculate the loss, regardless of the specific loss function chosen.
    *   Offering a method (``build_optimizer``) to construct an optimizer based on configuration parameters and associate it with model parameters. It manages the lifecycle of these optimizers, ensuring only one instance per type is created per ``lossfx`` object.
    *   Implementing various weight initialization methods via the ``weight_init`` function.
    *   Ensuring loss function modules are instantiated (``build_loss_function``) and moved to the correct device (``to`` method).
    *   Handling cleanup of all dynamically allocated optimizer and loss function objects in its destructor to prevent memory leaks.

    **Public Member Functions:**

    .. cpp:function:: lossfx()

        Constructs a new ``lossfx`` object.

        **Details:**

        This is the default constructor. It initializes the ``lossfx`` instance, setting all internal pointers that hold optimizer instances (e.g., ``m_adam``, ``m_sgd``) and loss function module instances (e.g., ``m_mse``, ``m_cross_entropy``) to ``nullptr``. This signifies that no optimizers or loss functions have been created yet by this ``lossfx`` object. It ensures a clean state upon object creation.

    .. cpp:function:: ~lossfx()

        Destroys the ``lossfx`` object and cleans up associated resources.

        **Details:**

        The destructor is crucial for preventing memory leaks. It systematically checks each internal pointer variable that might hold a dynamically allocated optimizer (like ``m_adam``, ``m_adagrad``, etc.) or a loss function module (like ``m_bce``, ``m_mse``, etc.). If a pointer is not ``nullptr``, it means an object was allocated during the lifetime of this ``lossfx`` instance (e.g., via ``build_optimizer`` or ``build_loss_function``). The destructor then calls ``delete`` on that pointer to free the associated memory. This ensures proper resource management for all objects created and managed by this class.

    .. cpp:function:: loss_enum loss_string(std::string name)

        Converts a string representation of a loss function name into its corresponding ``loss_enum`` value.

        **Details:**

        This function takes a string, converts it to lowercase using the inherited ``lower`` method (assumed from the ``tools`` base class), and compares it against a predefined set of known loss function names (e.g., "mse", "crossentropyloss", "l1", "huber"). The comparison is case-insensitive due to the initial lowercase conversion.

        :param name: The string identifier for the desired loss function (e.g., ``"MSE"``, ``"cross_entropy"``). Case does not matter.
        :returns: The ``loss_enum`` value corresponding to the matched name (e.g., ``loss_enum::mse``, ``loss_enum::cross_entropy``). If the input string ``name`` does not match any known loss function identifier, it returns ``loss_enum::invalid_loss`` to indicate failure.

    .. cpp:function:: opt_enum optim_string(std::string name)

        Converts a string representation of an optimizer name into its corresponding ``opt_enum`` value.

        **Details:**

        Similar to ``loss_string``, this function takes a string identifier for an optimizer, converts it to lowercase, and compares it against known optimizer names (e.g., "adam", "sgd", "rmsprop", "adamw"). The comparison is case-insensitive.

        :param name: The string identifier for the desired optimizer (e.g., ``"Adam"``, ``"SGD"``, ``"RMSProp"``). Case is ignored.
        :returns: The ``opt_enum`` value corresponding to the matched name (e.g., ``opt_enum::adam``, ``opt_enum::sgd``). If the input string ``name`` does not match any recognized optimizer identifier, it returns ``opt_enum::invalid_optimizer`` to signal an error.

    .. cpp:function:: torch::Tensor loss(torch::Tensor* pred, torch::Tensor* truth, loss_enum lss)

        Computes the loss between a set of predictions and corresponding ground truth values.

        **Details:**

        This function acts as the primary interface for loss calculation. It uses the provided ``loss_enum`` value (``lss``) to determine which specific loss function to apply. It first ensures the required loss function module has been instantiated (implicitly or explicitly via ``build_loss_function``). Then, it dispatches the calculation to one of the private overloaded ``_fx_loss`` helper methods based on the type of the instantiated loss function module associated with ``lss``. These helper methods handle the actual call to the loss function module's ``forward`` method.

        :param pred: A pointer to a ``torch::Tensor`` containing the output predictions generated by the model. The shape and type requirements depend on the specific loss function being used.
        :param truth: A pointer to a ``torch::Tensor`` containing the ground truth labels or target values corresponding to the predictions. Shape and type requirements also depend on the loss function.
        :param lss: An enum value of type ``loss_enum`` specifying which loss function should be used for the computation (e.g., ``loss_enum::mse``, ``loss_enum::cross_entropy``).
        :returns: A ``torch::Tensor`` containing the computed loss value. Typically, this is a scalar tensor (containing a single value), especially if reduction (like mean or sum) is applied within the loss function module. If the provided ``lss`` enum is invalid, or if the corresponding loss function is not implemented or fails, an empty tensor (``torch::Tensor()``) might be returned (as indicated by some ``_fx_loss`` implementations).

        .. warning::
            Ensure that the ``pred`` and ``truth`` tensors have compatible shapes, data types, and value ranges as expected by the chosen loss function (``lss``). Also ensure the required loss function module has been built using ``build_loss_function`` and moved to the correct device using ``to``.

    .. cpp:function:: void weight_init(torch::nn::Sequential* data, mlp_init method)

        Initializes the weights of layers within a ``torch::nn::Sequential`` module using a specified method.

        **Details:**

        This function iterates through all the modules contained within the provided ``torch::nn::Sequential`` container (``data``). For each module, it checks if it is a ``torch::nn::Linear`` layer. If it is, it applies the weight initialization strategy specified by the ``method`` parameter to the layer's weight tensor. Common methods include Xavier (Glorot) initialization (uniform or normal) and Kaiming (He) initialization (uniform or normal), often chosen based on the activation functions used in the network. This function likely uses static helper functions within the class or standard ``torch::nn::init`` functions to perform the actual initialization logic based on the ``method`` enum. Bias terms might also be initialized (e.g., to zero) depending on the implementation.

        :param data: A pointer to the ``torch::nn::Sequential`` module whose layers (specifically ``Linear`` layers) need their weights initialized.
        :param method: An enum value of type ``mlp_init`` specifying the desired weight initialization technique (e.g., ``mlp_init::xavier_uniform``, ``mlp_init::kaiming_normal``).

        .. note::
            This function typically modifies the weight tensors of the layers within the ``data`` module in-place.

    .. cpp:function:: torch::optim::Optimizer* build_optimizer(optimizer_params_t* op, std::vector<torch::Tensor>* params)

        Constructs (if necessary) and returns a pointer to a PyTorch optimizer instance.

        **Details:**

        This function serves as a factory for creating and retrieving optimizer objects. It takes the desired optimizer type and its configuration parameters via the ``op`` struct and the model parameters to be optimized via the ``params`` vector. Based on the ``optimizer_type`` specified in ``op``, it calls the corresponding private ``build_*`` helper function (e.g., ``build_adam``, ``build_sgd``). These helpers handle the actual instantiation using ``torch::optim::<OptimizerType>(params, options)``. Crucially, this function implements a singleton pattern per optimizer type *within this ``lossfx`` instance*. It checks if an optimizer of the requested type has already been created (i.e., if the corresponding member pointer like ``m_adam`` is not ``nullptr``). If it exists, it returns the existing pointer. If not, it creates the new optimizer instance, stores its pointer in the appropriate member variable, and then returns the pointer.

        :param op: A pointer to an ``optimizer_params_t`` struct. This struct must contain the desired optimizer type (``op->optimizer_type``) and all necessary hyperparameters (e.g., ``op->lr`` for learning rate, ``op->momentum``, ``op->weight_decay``, etc.) required by that specific optimizer.
        :param params: A pointer to a ``std::vector<torch::Tensor>`` containing all the model parameters (typically obtained via ``model->parameters()``) that the optimizer should manage and update.
        :returns: A pointer to the created or retrieved ``torch::optim::Optimizer`` object. The ownership of the memory pointed to remains with the ``lossfx`` object; the caller should *not* delete this pointer. Returns ``nullptr`` if the optimizer type specified in ``op`` is invalid (``opt_enum::invalid_optimizer``) or if the creation process fails for some reason (though standard LibTorch optimizers usually throw exceptions on failure rather than returning null).

        .. warning::
            The lifetime of the returned optimizer is tied to the lifetime of the ``lossfx`` object. Do not use the pointer after the ``lossfx`` object has been destroyed.

    .. cpp:function:: bool build_loss_function(loss_enum lss)

        Ensures that the PyTorch module for a specific loss function is instantiated.

        **Details:**

        This function checks if the loss function module corresponding to the given ``loss_enum`` (``lss``) has already been created and stored in its respective member pointer (e.g., ``m_mse`` for ``loss_enum::mse``). If the pointer is ``nullptr``, it proceeds to create a new instance of the appropriate ``torch::nn::<LossType>Impl`` class (e.g., ``torch::nn::MSELossImpl()``), potentially configuring it with default options or options derived from elsewhere (though configuration options are not explicitly passed here). The pointer to the newly created module is then stored in the corresponding member variable. This mechanism allows for lazy initialization – loss function modules are only created when they are first needed (either by calling ``loss`` or explicitly calling ``build_loss_function``).

        :param lss: The ``loss_enum`` value identifying the loss function module to build or ensure exists.
        :returns: ``true`` if the loss function module corresponding to ``lss`` already exists or was successfully created during this call. Returns ``false`` if the provided ``lss`` is ``loss_enum::invalid_loss`` or if the instantiation fails for any reason.

        .. note::
            The created loss function modules are managed by the ``lossfx`` object and will be deleted by its destructor.

    .. cpp:function:: void to(torch::TensorOptions* op)

        Moves all currently instantiated loss function modules to a specified device (CPU or GPU).

        **Details:**

        This function iterates through all the internal pointers that hold loss function modules (e.g., ``m_bce``, ``m_mse``, ``m_cross_entropy``, etc.). For each pointer that is not ``nullptr`` (meaning the corresponding loss module has been instantiated via ``build_loss_function``), it calls the module's ``to()`` method, passing the device specified in the ``torch::TensorOptions`` object pointed to by ``op``. This is essential for ensuring that the loss calculations occur on the same device as the model's output tensors and the target tensors, preventing device mismatch errors.

        :param op: A pointer to a ``torch::TensorOptions`` object. The device specified within these options (e.g., ``op->device()``, which could be ``torch::kCPU`` or ``torch::kCUDA``) determines the target device for the loss function modules. Other options within the struct (like dtype) are typically ignored by the module's ``to()`` method.

        .. note::
            This function should typically be called after creating the ``lossfx`` object and potentially building the necessary loss functions, and before starting the training loop, especially if training on a GPU. It modifies the loss function modules in-place.

    **Private Member Functions:**

    .. cpp:function:: void build_adam(optimizer_params_t* op, std::vector<torch::Tensor>* params)

        Internal helper function to create and configure an Adam optimizer instance.

        **Details:**

        Called exclusively by ``build_optimizer`` for ``opt_enum::adam``. Checks ``m_adam``. If ``nullptr``, creates ``torch::optim::Adam`` using hyperparameters from ``op`` (lr, beta1, beta2, eps, weight_decay, amsgrad) and ``params``. Stores pointer in ``m_adam``.

        :param op: Pointer to the ``optimizer_params_t`` struct containing configuration settings.
        :param params: Pointer to the vector of model parameters (``torch::Tensor``) to be optimized.

    .. cpp:function:: void build_adagrad(optimizer_params_t* op, std::vector<torch::Tensor>* params)

        Internal helper function to create and configure an Adagrad optimizer instance.

        **Details:**

        Called by ``build_optimizer`` for ``opt_enum::adagrad``. Checks ``m_adagrad``. If ``nullptr``, creates ``torch::optim::Adagrad`` using hyperparameters from ``op`` (lr, lr_decay, weight_decay, initial_accumulator_value, eps) and ``params``. Stores pointer in ``m_adagrad``.

        :param op: Pointer to the ``optimizer_params_t`` struct containing configuration settings.
        :param params: Pointer to the vector of model parameters (``torch::Tensor``) to be optimized.

    .. cpp:function:: void build_adamw(optimizer_params_t* op, std::vector<torch::Tensor>* params)

        Internal helper function to create and configure an AdamW optimizer instance.

        **Details:**

        Called by ``build_optimizer`` for ``opt_enum::adamw``. Checks ``m_adamw``. If ``nullptr``, creates ``torch::optim::AdamW`` using hyperparameters from ``op`` (lr, beta1, beta2, eps, weight_decay, amsgrad) and ``params``. Stores pointer in ``m_adamw``.

        :param op: Pointer to the ``optimizer_params_t`` struct containing configuration settings.
        :param params: Pointer to the vector of model parameters (``torch::Tensor``) to be optimized.

    .. cpp:function:: void build_lbfgs(optimizer_params_t* op, std::vector<torch::Tensor>* params)

        Internal helper function to create and configure an LBFGS optimizer instance.

        **Details:**

        Called by ``build_optimizer`` for ``opt_enum::lbfgs``. Checks ``m_lbfgs``. If ``nullptr``, creates ``torch::optim::LBFGS`` using hyperparameters from ``op`` (lr, max_iter, max_eval, tolerance_grad, tolerance_change, history_size) and ``params``. Stores pointer in ``m_lbfgs``.

        :param op: Pointer to the ``optimizer_params_t`` struct containing configuration settings.
        :param params: Pointer to the vector of model parameters (``torch::Tensor``) to be optimized.

        .. note::
            LBFGS requires a closure to re-evaluate the model and calculate loss, which is handled differently during the ``optimizer->step()`` call compared to first-order methods.

    .. cpp:function:: void build_rmsprop(optimizer_params_t* op, std::vector<torch::Tensor>* params)

        Internal helper function to create and configure an RMSprop optimizer instance.

        **Details:**

        Called by ``build_optimizer`` for ``opt_enum::rmsprop``. Checks ``m_rmsprop``. If ``nullptr``, creates ``torch::optim::RMSprop`` using hyperparameters from ``op`` (lr, alpha, eps, weight_decay, momentum, centered) and ``params``. Stores pointer in ``m_rmsprop``.

        :param op: Pointer to the ``optimizer_params_t`` struct containing configuration settings.
        :param params: Pointer to the vector of model parameters (``torch::Tensor``) to be optimized.

    .. cpp:function:: void build_sgd(optimizer_params_t* op, std::vector<torch::Tensor>* params)

        Internal helper function to create and configure an SGD optimizer instance.

        **Details:**

        Called by ``build_optimizer`` for ``opt_enum::sgd``. Checks ``m_sgd``. If ``nullptr``, creates ``torch::optim::SGD`` using hyperparameters from ``op`` (lr, momentum, dampening, weight_decay, nesterov) and ``params``. Stores pointer in ``m_sgd``.

        :param op: Pointer to the ``optimizer_params_t`` struct containing configuration settings.
        :param params: Pointer to the vector of model parameters (``torch::Tensor``) to be optimized.

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::BCELossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the Binary Cross Entropy loss function module.

        :param lossfx_: Pointer to an instantiated ``torch::nn::BCELossImpl`` module.
        :param pred: Pointer to the prediction tensor (probabilities, typically after sigmoid).
        :param truth: Pointer to the target tensor (binary values, 0 or 1).
        :returns: The calculated BCE loss tensor.

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::BCEWithLogitsLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the Binary Cross Entropy with Logits loss function module. More numerically stable than ``BCELoss`` preceded by a ``Sigmoid`` layer.

        :param lossfx_: Pointer to an instantiated ``torch::nn::BCEWithLogitsLossImpl`` module.
        :param pred: Pointer to the prediction tensor (raw logits, before sigmoid).
        :param truth: Pointer to the target tensor (binary values, 0 or 1).
        :returns: The calculated BCEWithLogits loss tensor.

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::CosineEmbeddingLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the Cosine Embedding loss function module.

        :param lossfx_: Pointer to an instantiated ``torch::nn::CosineEmbeddingLossImpl`` module.
        :param pred: Pointer to the first input tensor.
        :param truth: Pointer to the second input tensor (or target label tensor depending on usage).
        :returns: The calculated Cosine Embedding loss tensor.

        .. warning::
            The original comment suggests this might not be fully implemented (returns empty tensor).

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::CrossEntropyLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the Cross Entropy loss function module. Combines ``LogSoftmax`` and ``NLLLoss``. Suitable for multi-class classification.

        :param lossfx_: Pointer to an instantiated ``torch::nn::CrossEntropyLossImpl`` module.
        :param pred: Pointer to the prediction tensor (raw scores/logits for each class). Shape (N, C).
        :param truth: Pointer to the target tensor (class indices). Shape (N).
        :returns: The calculated Cross Entropy loss tensor.

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::CTCLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the Connectionist Temporal Classification loss function module. Used for sequence-to-sequence tasks like speech recognition where alignment is variable.

        :param lossfx_: Pointer to an instantiated ``torch::nn::CTCLossImpl`` module.
        :param pred: Pointer to the log-probabilities from the model output.
        :param truth: Pointer to the target sequences.
        :returns: The calculated CTC loss tensor.

        .. warning::
            The original comment suggests this might not be fully implemented (returns empty tensor). Requires specific input shapes and additional parameters (input lengths, target lengths).

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::HingeEmbeddingLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the Hinge Embedding loss function module. Measures loss for learning embeddings or semi-supervised learning.

        :param lossfx_: Pointer to an instantiated ``torch::nn::HingeEmbeddingLossImpl`` module.
        :param pred: Pointer to the input tensor.
        :param truth: Pointer to the target tensor containing labels (1 or -1).
        :returns: The calculated Hinge Embedding loss tensor.

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::HuberLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the Huber loss function module (Smooth L1 Loss variant). Less sensitive to outliers than MSELoss.

        :param lossfx_: Pointer to an instantiated ``torch::nn::HuberLossImpl`` module.
        :param pred: Pointer to the prediction tensor.
        :param truth: Pointer to the target tensor.
        :returns: The calculated Huber loss tensor.

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::KLDivLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the Kullback-Leibler Divergence loss function module. Measures the difference between two probability distributions.

        :param lossfx_: Pointer to an instantiated ``torch::nn::KLDivLossImpl`` module.
        :param pred: Pointer to the input tensor (log-probabilities).
        :param truth: Pointer to the target tensor (probabilities).
        :returns: The calculated KL Divergence loss tensor.

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::L1LossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the L1 loss function module (Mean Absolute Error).

        :param lossfx_: Pointer to an instantiated ``torch::nn::L1LossImpl`` module.
        :param pred: Pointer to the prediction tensor.
        :param truth: Pointer to the target tensor.
        :returns: The calculated L1 loss tensor.

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::MarginRankingLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the Margin Ranking loss function module. Used for ranking problems.

        :param lossfx_: Pointer to an instantiated ``torch::nn::MarginRankingLossImpl`` module.
        :param pred: Pointer to the first input tensor (or combined input).
        :param truth: Pointer to the second input tensor (or target tensor indicating relative rank).
        :returns: The calculated Margin Ranking loss tensor.

        .. warning::
            The original comment suggests this might not be fully implemented (returns empty tensor). Requires specific input setup (often two inputs and a target).

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::MSELossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the Mean Squared Error (L2) loss function module.

        :param lossfx_: Pointer to an instantiated ``torch::nn::MSELossImpl`` module.
        :param pred: Pointer to the prediction tensor.
        :param truth: Pointer to the target tensor.
        :returns: The calculated MSE loss tensor.

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::MultiLabelMarginLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the Multi-Label Margin loss function module. Suitable for multi-label classification problems (Hinge-based).

        :param lossfx_: Pointer to an instantiated ``torch::nn::MultiLabelMarginLossImpl`` module.
        :param pred: Pointer to the prediction tensor (scores for each class).
        :param truth: Pointer to the target tensor (containing indices of the true labels).
        :returns: The calculated Multi-Label Margin loss tensor.

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::MultiLabelSoftMarginLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the Multi-Label Soft Margin loss function module. Suitable for multi-label classification problems (Binary cross-entropy based).

        :param lossfx_: Pointer to an instantiated ``torch::nn::MultiLabelSoftMarginLossImpl`` module.
        :param pred: Pointer to the prediction tensor (scores/logits for each class).
        :param truth: Pointer to the target tensor (binary matrix indicating true labels).
        :returns: The calculated Multi-Label Soft Margin loss tensor.

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::MultiMarginLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the Multi-Margin loss function module. Hinge-based loss for multi-class classification.

        :param lossfx_: Pointer to an instantiated ``torch::nn::MultiMarginLossImpl`` module.
        :param pred: Pointer to the prediction tensor (scores for each class).
        :param truth: Pointer to the target tensor (class indices).
        :returns: The calculated Multi-Margin loss tensor.

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::NLLLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the Negative Log Likelihood loss function module. Useful when input is log-probabilities.

        :param lossfx_: Pointer to an instantiated ``torch::nn::NLLLossImpl`` module.
        :param pred: Pointer to the prediction tensor (log-probabilities). Shape (N, C).
        :param truth: Pointer to the target tensor (class indices). Shape (N).
        :returns: The calculated NLL loss tensor.

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::PoissonNLLLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the Poisson Negative Log Likelihood loss function module. Suitable for count targets.

        :param lossfx_: Pointer to an instantiated ``torch::nn::PoissonNLLLossImpl`` module.
        :param pred: Pointer to the prediction tensor (expected counts, often model output).
        :param truth: Pointer to the target tensor (observed counts).
        :returns: The calculated Poisson NLL loss tensor.

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::SmoothL1LossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the Smooth L1 loss function module. Combination of L1 and L2 loss.

        :param lossfx_: Pointer to an instantiated ``torch::nn::SmoothL1LossImpl`` module.
        :param pred: Pointer to the prediction tensor.
        :param truth: Pointer to the target tensor.
        :returns: The calculated Smooth L1 loss tensor.

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::SoftMarginLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the Soft Margin loss function module. Optimizes a two-class classification logistic loss.

        :param lossfx_: Pointer to an instantiated ``torch::nn::SoftMarginLossImpl`` module.
        :param pred: Pointer to the prediction tensor (scores/logits).
        :param truth: Pointer to the target tensor (labels as 1 or -1).
        :returns: The calculated Soft Margin loss tensor.

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::TripletMarginLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the Triplet Margin loss function module. Used for learning embeddings.

        :param lossfx_: Pointer to an instantiated ``torch::nn::TripletMarginLossImpl`` module.
        :param pred: Pointer to the anchor input tensor.
        :param truth: Pointer to the positive/negative input tensors (requires specific input setup).
        :returns: The calculated Triplet Margin loss tensor.

        .. warning::
            The original comment suggests this might not be fully implemented (returns empty tensor). Requires three inputs: anchor, positive, and negative samples.

    .. cpp:function:: torch::Tensor _fx_loss(torch::nn::TripletMarginWithDistanceLossImpl* lossfx_, torch::Tensor* pred, torch::Tensor* truth)

        Calculates loss using the Triplet Margin with Distance loss function module. Allows specifying a custom distance function.

        :param lossfx_: Pointer to an instantiated ``torch::nn::TripletMarginWithDistanceLossImpl`` module.
        :param pred: Pointer to the anchor input tensor.
        :param truth: Pointer to the positive/negative input tensors (requires specific input setup).
        :returns: The calculated Triplet Margin with Distance loss tensor.

        .. warning::
            The original comment suggests this might not be fully implemented (returns empty tensor). Requires three inputs and potentially a distance function.

    **Private Member Variables:**

    *Optimizers (pointers managed by this class)*

    .. cpp:var:: torch::optim::Adam* m_adam

        Pointer holding the single instance of the Adam optimizer, if created. Managed by ``build_optimizer`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::optim::Adagrad* m_adagrad

        Pointer holding the single instance of the Adagrad optimizer, if created. Managed by ``build_optimizer`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::optim::AdamW* m_adamw

        Pointer holding the single instance of the AdamW optimizer, if created. Managed by ``build_optimizer`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::optim::LBFGS* m_lbfgs

        Pointer holding the single instance of the LBFGS optimizer, if created. Managed by ``build_optimizer`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::optim::RMSprop* m_rmsprop

        Pointer holding the single instance of the RMSprop optimizer, if created. Managed by ``build_optimizer`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::optim::SGD* m_sgd

        Pointer holding the single instance of the SGD optimizer, if created. Managed by ``build_optimizer`` and the destructor. Initialized to ``nullptr``.

    *Loss Functions (pointers managed by this class)*

    .. cpp:var:: torch::nn::BCELossImpl* m_bce

        Pointer holding the instance of the BCELoss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::nn::BCEWithLogitsLossImpl* m_bce_with_logits

        Pointer holding the instance of the BCEWithLogitsLoss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::nn::CosineEmbeddingLossImpl* m_cosine_embedding

        Pointer holding the instance of the CosineEmbeddingLoss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::nn::CrossEntropyLossImpl* m_cross_entropy

        Pointer holding the instance of the CrossEntropyLoss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::nn::CTCLossImpl* m_ctc

        Pointer holding the instance of the CTCLoss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::nn::HingeEmbeddingLossImpl* m_hinge_embedding

        Pointer holding the instance of the HingeEmbeddingLoss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::nn::HuberLossImpl* m_huber

        Pointer holding the instance of the HuberLoss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::nn::KLDivLossImpl* m_kl_div

        Pointer holding the instance of the KLDivLoss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::nn::L1LossImpl* m_l1

        Pointer holding the instance of the L1Loss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::nn::MarginRankingLossImpl* m_margin_ranking

        Pointer holding the instance of the MarginRankingLoss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::nn::MSELossImpl* m_mse

        Pointer holding the instance of the MSELoss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::nn::MultiLabelMarginLossImpl* m_multi_label_margin

        Pointer holding the instance of the MultiLabelMarginLoss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::nn::MultiLabelSoftMarginLossImpl* m_multi_label_soft_margin

        Pointer holding the instance of the MultiLabelSoftMarginLoss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::nn::MultiMarginLossImpl* m_multi_margin

        Pointer holding the instance of the MultiMarginLoss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::nn::NLLLossImpl* m_nll

        Pointer holding the instance of the NLLLoss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::nn::PoissonNLLLossImpl* m_poisson_nll

        Pointer holding the instance of the PoissonNLLLoss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::nn::SmoothL1LossImpl* m_smooth_l1

        Pointer holding the instance of the SmoothL1Loss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::nn::SoftMarginLossImpl* m_soft_margin

        Pointer holding the instance of the SoftMarginLoss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::nn::TripletMarginLossImpl* m_triplet_margin

        Pointer holding the instance of the TripletMarginLoss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.

    .. cpp:var:: torch::nn::TripletMarginWithDistanceLossImpl* m_triplet_margin_with_distance

        Pointer holding the instance of the TripletMarginWithDistanceLoss module, if created. Managed by ``build_loss_function`` and the destructor. Initialized to ``nullptr``.


.. cpp:class:: optimizer : public tools, public notification

    Manages the training, validation, and evaluation processes for machine learning models.

    This class orchestrates the different phases of model development, potentially using k-fold
    cross-validation. It handles data loading, model configuration, execution of training/validation/evaluation
    loops, metric collection, and reporting. It inherits from ``tools`` and ``notification`` for utility
    functions and messaging capabilities.

    .. cpp:member:: settings_t m_settings

        General settings for the optimization process.

        Holds configuration parameters like number of epochs, batch size, run name,
        flags for enabling training, validation, evaluation, debug mode, etc.

    .. cpp:function:: optimizer()

        Default constructor for the optimizer class.

        Initializes the notification prefix to "optimizer" and allocates a new ``metrics`` object.

    .. cpp:function:: ~optimizer()

        Destructor for the optimizer class.

        Deallocates the ``metrics`` object and cleans up all ``model_template`` instances
        stored in the ``kfold_sessions`` map to prevent memory leaks.

    .. cpp:function:: void import_dataloader(dataloader* dl)

        Imports a dataloader instance into the optimizer.

        :param dl: A pointer to the ``dataloader`` object providing access to datasets.

        Sets the settings for the internal ``metrics`` object based on ``m_settings`` and
        stores the pointer ``dl`` for later use in training/validation/evaluation loops.

    .. cpp:function:: void import_model_sessions(std::tuple<model_template*, optimizer_params_t*>* models)

        Imports the base model template and its associated optimizer parameters.

        :param models: A pointer to a tuple containing:
                            - A pointer to the base ``model_template`` to be used.
                            - A pointer to the ``optimizer_params_t`` configuration for the model's optimizer.

        This function configures the base model, clones it, sets up its optimizer,
        performs initial validation checks using random samples from the dataloader,
        and ensures the model is ready for the training process.

    .. cpp:function:: void training_loop(int k, int epoch)

        Executes a single training epoch for a specific k-fold split.

        :param k: The index of the current k-fold split (0-based).
        :param epoch: The current epoch number (0-based).

        Retrieves the training dataset for fold ``k``, sets the corresponding model to training mode
        (enabling gradients and dropout), optionally batches the data, iterates through the samples,
        performs the forward pass and backpropagation, captures training metrics, and saves the model's state.

    .. cpp:function:: void validation_loop(int k, int epoch)

        Executes a single validation epoch for a specific k-fold split.

        :param k: The index of the current k-fold split (0-based).
        :param epoch: The current epoch number (0-based).

        Retrieves the validation dataset for fold ``k``, sets the corresponding model to evaluation mode
        (disabling gradients and dropout), optionally batches the data, iterates through the samples
        within a ``torch::NoGradGuard`` block, performs the forward pass, and captures validation metrics.

    .. cpp:function:: void evaluation_loop(int k, int epoch)

        Executes a single evaluation epoch using the test set for a specific k-fold split's model.

        :param k: The index of the k-fold split whose model is being evaluated (0-based).
        :param epoch: The current epoch number (0-based), often the final epoch after training.

        Retrieves the test dataset, sets the model for fold ``k`` to evaluation mode, optionally batches
        the data, iterates through the samples within a ``torch::NoGradGuard`` block, performs the
        forward pass, and captures evaluation metrics.

    .. cpp:function:: void launch_model(int k)

        Launches the complete training/validation/evaluation cycle for a specific k-fold split.

        :param k: The index of the k-fold split to process (0-based).

        Iterates through the configured number of epochs (``m_settings.epochs``). In each epoch,
        it calls ``training_loop``, ``validation_loop``, and ``evaluation_loop`` based on the flags
        set in ``m_settings``. It handles potential debugging plot generation and ensures plotting
        threads complete before proceeding. Finally, marks the corresponding model report as complete.

    .. cpp:member:: std::map<int, model_template*> kfold_sessions = {}

        Map storing the model instances for each k-fold split.

        The key is the integer index ``k`` of the fold, and the value is a pointer
        to the ``model_template`` instance used for that fold.

    .. cpp:member:: std::map<std::string, model_report*> reports = {}

        Map storing reports generated during the optimization process.

        The key is typically a string combining the run name and the fold index (e.g., "run1_0").
        The value is a pointer to a ``model_report`` object containing metrics and status for that run/fold.

    .. cpp:member:: metrics* metric = nullptr

        Pointer to the metrics collection object.

        Used to record performance metrics (loss, accuracy, etc.) during different phases.

    .. cpp:member:: dataloader* loader = nullptr

        Pointer to the dataloader object.

        Provides access to training, validation, and test datasets.

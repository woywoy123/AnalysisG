.. cpp:file:: analysis.h

.. highlight:: cpp

.. cpp:function:: template <typename g> void flush(std::map<std::string, g*>* data)

    Template function to safely delete dynamically allocated objects stored as values in a map.

    Iterates through the map and calls delete on each pointer stored as a value.
    Finally, clears the map.

    :tparam g: The type of the pointer stored in the map (e.g., optimizer*, model_template*).
    :param data: Pointer to the map whose values need to be deleted and cleared.

.. cpp:class:: analysis : public notification, public tools

    Main analysis class orchestrating the entire workflow.

    This class manages the configuration, data loading, event building,
    graph generation, model training, inference, metric calculation, and
    selection application. It utilizes helper classes like ``io``, ``sampletracer``,
    ``dataloader``, and various template classes (``event_template``, ``graph_template``, etc.)
    to perform these tasks. It inherits from ``notification`` for logging and ``tools``
    for utility functions.

    .. cpp:function:: analysis()

        Constructor for the analysis class.

        Initializes internal data structures (vectors, maps, pointers).
        Sets up necessary paths, including the dictionary path for ROOT PCM files.
        Checks if PCM files need building and configures ROOT's build environment accordingly.
        Builds essential dictionaries (meta_t, weights_t) and potentially others in parallel threads.
        Resets ROOT's working directory after dictionary building.

    .. cpp:function:: ~analysis()

        Destructor for the analysis class.

        Cleans up dynamically allocated resources.
        Flushes data associated with k-folds (``tags``).
        Deletes objects stored in maps (``trainer``, ``model_metrics``, ``metric_names``).
        Deletes the ``dataloader``, ``sampletracer``, and ``io`` reader objects.

    .. cpp:function:: void add_samples(std::string path, std::string label)

        Adds a sample dataset path and associates it with a label.

        The path can point to a directory containing ROOT files or a single ROOT file.
        This label is used later to associate event/graph templates with specific samples.

        :param path: The path to the sample dataset (directory or file).
        :param label: A string label to identify this sample group.

    .. cpp:function:: void add_selection_template(selection_template* sel)

        Adds a selection template to the analysis workflow.

        Selection templates define criteria to filter or categorize events.

        :param sel: Pointer to a selection_template object. The analysis class
                        does not take ownership of this pointer directly but might clone it.

    .. cpp:function:: void add_event_template(event_template* ev, std::string label)

        Adds an event template and associates it with a sample label.

        Event templates define how to read data from ROOT files and construct event objects.

        :param ev: Pointer to an event_template object.
        :param label: The sample label this template should be applied to.

    .. cpp:function:: void add_graph_template(graph_template* gr, std::string label)

        Adds a graph template and associates it with a sample label.

        Graph templates define how to convert event objects into graph structures
        suitable for GNNs. A single sample label can have multiple graph types associated.

        :param gr: Pointer to a graph_template object.
        :param label: The sample label this template should be applied to.

    .. cpp:function:: void add_metric_template(metric_template* mx, model_template* mdl)

        Adds a metric template and links it to a specific model template.

        Metric templates define how to calculate performance metrics based on model outputs.
        The metric is linked to a model to ensure compatibility and access to model predictions.
        Clones the metric and model templates internally if they haven't been added before.

        :param mx: Pointer to a metric_template object.
        :param mdl: Pointer to a model_template object that this metric will evaluate.

    .. cpp:function:: void add_model(model_template* model, optimizer_params_t* op, std::string run_name)

        Adds a model configuration for a training session.

        Associates a model template with specific optimizer parameters and a unique run name
        for a training session.

        :param model: Pointer to a model_template object to be trained.
        :param op: Pointer to an optimizer_params_t object containing training hyperparameters.
        :param run_name: A unique name for this training run/session.

    .. cpp:function:: void add_model(model_template* model, std::string run_name)

        Adds a model configuration for an inference session.

        Associates a model template with a unique run name for inference. The model state
        is expected to be loaded from a checkpoint later.

        :param model: Pointer to a model_template object to be used for inference.
        :param run_name: A unique name for this inference run.

    .. cpp:function:: void attach_threads()

        Waits for all background threads (e.g., training loops) to complete.

        Joins all threads stored in the internal ``threads`` vector.

    .. cpp:function:: void start()

        Starts the main analysis workflow.

        This is the primary execution function. It orchestrates the entire process:
        1. Sets up logging and thread count based on settings.
        2. Fetches k-fold tags if pre-tagging is enabled.
        3. Creates the main output directory.
        4. Checks the cache for existing graph files.
        5. Builds event objects from ROOT files (if not skipped due to cache).
        6. Scans ROOT file metadata if no event/graph templates were provided.
        7. Enables ROOT's implicit multi-threading.
        8. Builds selections (if defined).
        9. Builds graphs from events (if defined and not cached).
        10. Compiles event/graph objects in the tracer.
        11. Fills selection information (if selections were built).
        12. Populates the dataloader.
        13. Dumps graphs to cache (if enabled).
        14. Restores graphs from cache (if cache path provided and building is skipped).
        15. Builds and executes metric calculations (if defined).
        16. Sets up and starts model training sessions (if defined).
        17. Sets up and starts model inference (if defined).

    .. cpp:function:: std::map<std::string, std::vector<float>> progress()

        Retrieves the progress of ongoing training sessions.

        :return: A map where keys are unique run identifiers (run_name + kfold index)
                    and values are vectors containing [progress percentage, current iterations, total events].

    .. cpp:function:: std::map<std::string, std::string> progress_mode()

        Retrieves the current operational mode of ongoing training sessions.

        :return: A map where keys are unique run identifiers and values are strings
                    describing the current mode (e.g., "Training|k-1|RunName: MyRun|Epoch: 5").

    .. cpp:function:: std::map<std::string, std::string> progress_report()

        Retrieves formatted progress reports from ongoing training sessions.

        Also triggers the dumping of any pending plots associated with the reports.

        :return: A map where keys are unique run identifiers and values are formatted
                    strings containing metrics like loss, accuracy, etc.

    .. cpp:function:: std::map<std::string, bool> is_complete()

        Checks if the training sessions have completed.

        :return: A map where keys are unique run identifiers and values are booleans
                    indicating whether the corresponding session is complete.

    .. cpp:member:: settings_t m_settings

        Global analysis settings.
        Configurable parameters controlling various aspects like paths, threading, debugging, etc.

    .. cpp:member:: std::map<std::string, meta*> meta_data

        Map storing metadata associated with input ROOT files.
        Keys are file paths, values are pointers to meta objects containing information
        like cross-sections, sum of weights, k-fold assignments, etc.

    private:

    .. cpp:function:: void check_cache()

        Checks for existing graph cache files (.h5).

        Compares the list of input ROOT files and defined graph templates against
        files found in the specified graph cache directory. Populates internal maps
        (``in_cache``, ``skip_event_build``) to indicate which graphs are already cached
        and whether event building can be skipped for certain input files.
        Updates ``file_labels`` to use absolute paths.

    .. cpp:function:: void build_project()

        Sets up the output directory structure for the analysis project.

        Creates the main output path specified in ``m_settings``.
        Defines and potentially creates subdirectories for model checkpoints and metric results
        based on the added models and metrics.

    .. cpp:function:: void build_events()

        Builds event objects from input ROOT files using event templates.

        Determines which files need processing based on ``skip_event_build``.
        Scans necessary trees/branches/leaves defined by the relevant event template.
        Launches multiple threads (based on ``m_settings.threads``) to read ROOT files
        and construct event objects in parallel using a lambda function.
        Stores the resulting event objects and associated metadata in the ``sampletracer``.
        Uses a progress bar to monitor the process.

    .. cpp:function:: void build_selections()

        Applies selection templates to the built event objects.

        Retrieves all events from the ``sampletracer``.
        Iterates through each registered selection template and applies its ``build`` method
        to every event.
        Adds the resulting selection information (e.g., pass/fail flags) to the ``sampletracer``.
        Optionally writes selection results to ROOT files if ``m_settings.selection_root`` is true.

    .. cpp:function:: void build_graphs()

        Builds graph objects from event objects using graph templates.

        Iterates through sample labels and their associated graph templates.
        Retrieves events corresponding to the current label from the ``sampletracer``.
        For each event and applicable graph template, checks if the graph exists in the cache (``in_cache``).
        If not cached, calls the ``build`` method of the graph template to create a graph object.
        Adds the resulting graph object to the ``sampletracer``.

    .. cpp:function:: void build_model_session()

        Sets up and launches model training sessions.

        Skips if no models were added for training.
        Determines the k-folds to run based on settings or defaults.
        Transfers required data (graphs) to the target devices (CPU/GPU) using the ``dataloader``.
        For each model session added via ``add_model``:
          - Creates an ``optimizer`` instance.
          - Configures the optimizer with settings and the dataloader.
          - Imports the model template and optimizer parameters.
          - Launches training loops for each specified k-fold using ``initialize_loop``.
          - Training loops can run in separate threads (default) or sequentially (debug mode).
          - Stores ``model_report`` objects to track progress.

    .. cpp:function:: void build_inference()

        Sets up and executes model inference.

        Retrieves the inference dataset from the ``dataloader``.
        Sorts data and prepares for multi-threaded execution across available models and samples.
        Transfers data to target devices.
        Disables ROOT error messages during inference.
        Launches multiple threads using the ``execution`` helper function. Each thread handles
        inference for a specific model on a subset of the data.
        Manages batching if ``m_settings.batch_size > 1``.
        Prepares output ROOT file structure and variable definitions using ``add_content``.
        Monitors thread execution using a progress bar.
        Cleans up temporary data structures after completion.

    .. cpp:function:: bool build_metric()

        Prepares and executes metric calculations.

        Skips if no metrics were added.
        Checks if a dataset is available in the ``dataloader``.
        Determines the required devices (CPU/GPU) for all metrics.
        Ensures the dataloader is populated and transfers data to the required devices.
        Builds batches of data (training, validation, test sets) for each unique combination
        of device, k-fold, and model needed by the metrics, caching batches where possible.
        Starts the CUDA server if necessary.
        Sets up the output directory structure via ``build_project``.
        Creates a list of ``metric_t`` tasks to be executed.
        Remaps tasks to distribute them across devices efficiently.
        Launches multiple threads using the ``execution_metric`` helper function to compute metrics.
        Monitors thread execution using a progress bar.
        Cleans up batched data after completion.

        :return: True if metrics were processed successfully or none were defined, false on failure (e.g., no data).

    .. cpp:function:: void build_metric_folds()

        Determines the set of k-folds required by all registered metrics.

        Iterates through all added metric templates, retrieves the k-folds they operate on,
        and compiles a unique list of all required k-folds. Stores this list in ``m_settings.kfold``.

    .. cpp:function:: void build_dataloader(bool training)

        Populates the dataloader with graph data and prepares train/test/validation splits.

        If the dataloader is empty, populates it with graphs from the ``sampletracer``.
        If ``training`` is true:
          - Tries to restore a pre-split dataset from ``m_settings.training_dataset``.
          - If restoration fails or no path is given, generates new train/test splits
             (using ``m_settings.train_size``) and k-fold splits (using ``m_settings.kfolds``).
          - If a path is given, dumps the newly generated dataset splits to that path.

        :param training: If true, generate/restore training, validation, and k-fold sets.

    .. cpp:function:: void fetchtags()

        Fetches pre-assigned k-fold information from a dataset file.

        This is called if ``m_settings.pretagevents`` is true.
        Reads ``folds_t`` objects from the file specified in ``m_settings.training_dataset``
        under the key "kfolds" and stores them in the ``tags`` vector.

    .. cpp:member:: bool started

        Flag indicating if the ``start()`` method has been called at least once.

    .. cpp:function:: static int add_content(std::map<std::string, torch::Tensor*>* data, std::vector<variable_t>* content, int index, std::string prefx, TTree* tt = nullptr)

        Static helper function to add tensor data as branches to a TTree.

        Iterates through the input map of tensors. For each tensor, it creates
        a ``variable_t`` object (which handles the TTree branching and filling logic)
        and adds it to the ``content`` vector. If ``tt`` is provided, it attempts to
        create the corresponding branch in the TTree immediately.

        :param data: Map of string names to torch::Tensor pointers.
        :param content: Vector of ``variable_t`` objects to store branch information.
        :param index: Starting index in the ``content`` vector.
        :param prefx: Prefix to add to the tensor names when creating branch names.
        :param tt: Optional pointer to the TTree to add branches to.
        :return: The updated index after adding variables.

    .. cpp:function:: static void add_content(std::map<std::string, torch::Tensor*>* data, std::vector<std::vector<torch::Tensor>>* buff, torch::Tensor* edge, torch::Tensor* node, torch::Tensor* batch, std::vector<long> mask)

        Static helper function to extract and buffer tensor data for specific events within a batch.

        Used during inference output writing. It takes batched tensors (graph, node, edge features)
        and extracts the parts corresponding to individual events specified by the ``mask``.
        The extracted tensors for each event are appended to the corresponding vector in ``buff``.
        It determines which index tensor (edge, node, or batch) to use based on the tensor's first dimension size.

        :param data: Map of feature names to batched torch::Tensor pointers.
        :param buff: Vector of vectors, where each inner vector will store the tensors for one event.
        :param edge: Batched edge index tensor.
        :param node: Batched node feature tensor (or similar tensor indexed by node).
        :param batch: Batched batch index tensor (mapping nodes/edges to events).
        :param mask: Vector of event indices within the batch to extract data for.

    .. cpp:function:: static void execution(model_template* mdx, model_settings_t mds, std::vector<graph_t*>* data, size_t* prg, std::string output, std::vector<variable_t>* content, std::string* msg)

        Static helper function executed in a thread to perform model inference and save results.

        Clones the model template, restores its state, and sets it to evaluation mode.
        Checks if the output file already exists and contains the expected number of entries; if so, skips execution.
        Creates/overwrites the output ROOT file and TTree.
        Iterates through the provided data (vector of graph pointers, potentially batches).
        Performs model forward pass (``md->forward``).
        Extracts input and output tensors for each event in the batch using ``add_content``.
        Initializes TTree branches using ``add_content`` on the first batch.
        Fills the TTree for each event.
        Updates the progress counter ``prg``.
        Updates the status message ``msg``.
        Cleans up the cloned model and closes the ROOT file.

        :param mdx: Pointer to the original model template.
        :param mds: Model settings structure.
        :param data: Vector of graph_t pointers (can be single graphs or batches).
        :param prg: Pointer to the progress counter for this thread.
        :param output: Path to the output ROOT file.
        :param content: Vector defining the variables/branches to be saved in the TTree.
        :param msg: Pointer to a string for status messages/logging.

    .. cpp:function:: static void execution_metric(metric_t* mt, size_t* prg, std::string* msg)

        Static helper function executed in a thread to perform metric calculation.

        Clones the metric template.
        Calls the ``execute`` method of the original metric template, passing the cloned
        instance and other necessary information. The actual metric logic resides within
        the ``metric_template::execute`` implementation.
        Cleans up the cloned metric template.

        :param mt: Pointer to the metric_t task object containing metric details and data links.
        :param prg: Pointer to the progress counter for this thread.
        :param msg: Pointer to a string for status messages/logging.

    .. cpp:function:: static void initialize_loop(optimizer* op, int k, model_template* model, optimizer_params_t* config, model_report** rep)

        Static helper function executed in a thread to initialize and manage a model training loop for one k-fold.

        Sets the CUDA device if applicable.
        Clones the model template.
        Sets up the optimizer and initializes the model.
        Determines the starting epoch, potentially resuming from a checkpoint if ``continue_training`` is enabled.
        Performs a check pass with a random data sample.
        Registers the model with the optimizer and metric reporting system.
        Stores the cloned model in the optimizer's session map.
        Creates a ``model_report`` object and assigns it to the output parameter ``rep``.
        Launches the actual training loop via ``op->launch_model(k)``.

        :param op: Pointer to the optimizer managing this training session.
        :param k: The k-fold index (0-based).
        :param model: Pointer to the original model template.
        :param config: Pointer to the optimizer configuration parameters.
        :param rep: Output pointer to store the created model_report object.

    .. cpp:function:: template <typename g> void safe_clone(std::map<std::string, g*>* mp, g* in)

        Safely clones a template object into a map if it doesn't already exist.

        Checks if a key ``in->name`` exists in the map ``mp``. If not, it clones the
        input object ``in`` using its ``clone(1)`` method and inserts the clone into the map.

        :tparam g: The type of the object to be cloned (e.g., model_template, metric_template).
        :param mp: Pointer to the map where the clone should be stored.
        :param in: Pointer to the object to be potentially cloned.

    .. cpp:member:: std::map<std::string, std::string> file_labels

        Maps input sample paths (files/directories) to their assigned labels.

    .. cpp:member:: std::map<std::string, event_template*> event_labels

        Maps sample labels to their corresponding event templates.

    .. cpp:member:: std::map<std::string, metric_template*> metric_names

        Maps metric names to their corresponding metric templates.

    .. cpp:member:: std::map<std::string, selection_template*> selection_names

        Maps selection names to their corresponding selection templates.

    .. cpp:member:: std::map<std::string, std::map<std::string, graph_template*>> graph_labels

        Maps sample labels to a map of graph names and their graph templates.

    .. cpp:member:: std::vector<std::string> model_session_names

        Stores the run names for model training sessions.

    .. cpp:member:: std::map<std::string, model_template*> model_inference

        Maps run names to model templates configured for inference.

    .. cpp:member:: std::map<std::string, model_template*> model_metrics

        Maps model names to model templates used by metrics.

    .. cpp:member:: std::vector<std::tuple<model_template*, optimizer_params_t*>> model_sessions

        Stores tuples of model templates and optimizer parameters for training sessions.

    .. cpp:member:: std::map<std::string, optimizer*> trainer

        Maps run names to optimizer instances managing training sessions.

    .. cpp:member:: std::map<std::string, model_report*> reports

        Maps unique run identifiers (run_name + kfold) to model report objects for progress tracking.

    .. cpp:member:: std::vector<std::thread*> threads

        Vector storing pointers to active background threads (e.g., training loops).

    .. cpp:member:: std::map<std::string, std::map<std::string, bool>> in_cache

        Nested map indicating cache status: ``in_cache[root_file][graph_type/hashed_name.h5] = true/false``.

    .. cpp:member:: std::map<std::string, bool> skip_event_build

        Map indicating whether event building can be skipped for a ROOT file (if all its graphs are cached).

    .. cpp:member:: std::map<std::string, std::string> graph_types

        Map storing unique graph type names encountered.

    .. cpp:member:: std::vector<folds_t>* tags

        Pointer to a vector storing k-fold assignment data (``folds_t``). Loaded if ``pretagevents`` is true.

    .. cpp:member:: dataloader* loader

        Pointer to the data loader instance, managing datasets and batching.

    .. cpp:member:: sampletracer* tracer

        Pointer to the sample tracer instance, holding intermediate event/graph/selection objects.

    .. cpp:member:: io* reader

        Pointer to the IO instance, used primarily for initial ROOT file scanning.

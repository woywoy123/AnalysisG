.. cpp:struct:: analytics_t

    Holds all analytics data, primarily histograms and configuration, for a single k-fold iteration.

    This structure serves as a container for all data related to performance monitoring
    for one specific fold in a k-fold cross-validation setup. It aggregates pointers
    to the model being evaluated, the final performance report, the current epoch state,
    and various maps storing TH1F histograms. These histograms track metrics like loss
    and accuracy, categorized by operational mode (training, validation, evaluation)
    and the part of the graph they relate to (graph, node, edge). It also includes
    histograms for invariant mass reconstruction if configured.

    .. cpp:member:: model_template* model = nullptr

        Pointer to the machine learning model instance associated with this k-fold.

    .. cpp:member:: model_report* report = nullptr

        Pointer to the structure that will store the final summarized performance metrics for this k-fold.

    .. cpp:member:: int this_epoch = 0

        Stores the current epoch number being processed or recorded for this k-fold.

    .. cpp:member:: std::map<mode_enum, std::map<std::string, TH1F*>> loss_graph = {}

        Map storing loss histograms for graph-level predictions. Keyed by [Mode][VariableName].

    .. cpp:member:: std::map<mode_enum, std::map<std::string, TH1F*>> loss_node = {}

        Map storing loss histograms for node-level predictions. Keyed by [Mode][VariableName].

    .. cpp:member:: std::map<mode_enum, std::map<std::string, TH1F*>> loss_edge = {}

        Map storing loss histograms for edge-level predictions. Keyed by [Mode][VariableName].

    .. cpp:member:: std::map<mode_enum, std::map<std::string, TH1F*>> accuracy_graph = {}

        Map storing accuracy histograms for graph-level predictions. Keyed by [Mode][VariableName].

    .. cpp:member:: std::map<mode_enum, std::map<std::string, TH1F*>> accuracy_node = {}

        Map storing accuracy histograms for node-level predictions. Keyed by [Mode][VariableName].

    .. cpp:member:: std::map<mode_enum, std::map<std::string, TH1F*>> accuracy_edge = {}

        Map storing accuracy histograms for edge-level predictions. Keyed by [Mode][VariableName].

    .. cpp:member:: std::map<mode_enum, std::map<std::string, TH1F*>> pred_mass_edge = {}

        Map storing predicted invariant mass histograms derived from edge predictions. Keyed by [Mode][VariableName].

    .. cpp:member:: std::map<mode_enum, std::map<std::string, TH1F*>> truth_mass_edge = {}

        Map storing true invariant mass histograms derived from ground truth edge information. Keyed by [Mode][VariableName].

    .. cpp:function:: void purge()

        Releases resources held by this analytics_t instance.

        This function is responsible for cleaning up dynamically allocated memory
        associated with this k-fold's analytics. It iterates through all the histogram
        maps (loss, accuracy, mass) and calls the ``destroy`` helper function to delete
        each individual TH1F object. It also deletes the associated ``model_report`` object.

    .. cpp:function:: void destroy(std::map<mode_enum, std::map<std::string, TH1F*>>* data)

        Helper function to recursively delete histograms stored within nested maps.

        Iterates through the provided nested map structure. The outer map is keyed by
        ``mode_enum`` (training, validation, evaluation), and the inner map is keyed by
        a string representing the variable name. Each value in the inner map is a
        pointer to a TH1F histogram object, which is deleted using ``delete``.

        :param[in,out] data: Pointer to a map of maps (``std::map<mode_enum, std::map<std::string, TH1F*>>*``)
                                    containing the TH1F pointers to be deleted. The map itself is not deleted,
                                    but its contents (the histograms) are.


.. cpp:class:: metrics :public: tools, public: notification

    Manages the calculation, storage, and visualization of performance metrics for machine learning models.

    This class provides a comprehensive framework for handling performance metrics throughout
    the lifecycle of model training and evaluation, particularly designed for k-fold cross-validation.
    It allows registering models for specific folds, capturing performance data (loss, accuracy,
    and optionally reconstructed invariant mass) during different operational modes (training,
    validation, evaluation), storing this time-series data in ROOT TH1F histograms, and generating
    various plots (e.g., loss/accuracy vs. epoch, mass distributions) to visualize the results.

    The class inherits from ``tools`` (presumably providing utility functions like directory creation)
    and ``notification`` (likely for logging messages and status updates). It maintains a registry
    of ``analytics_t`` structures, one for each k-fold being managed.

    **Public Members:**

    .. cpp:function:: metrics()

        Constructs the metrics manager object.

        Initializes the metrics system. As part of the initialization, it sets the
        global ROOT error message ignore level to 3000 (kError) using ``gErrorIgnoreLevel``.
        This suppresses informational and warning messages from ROOT, focusing only on errors.

    .. cpp:function:: ~metrics()

        Destroys the metrics manager object and cleans up associated resources.

        Ensures proper cleanup of all allocated resources. It iterates through the
        ``registry`` map, which contains ``analytics_t`` objects for each k-fold. For each
        ``analytics_t`` object, it calls the ``purge()`` method to delete all associated
        histograms and the ``model_report``. Finally, it clears the ``registry`` map itself.

    .. cpp:member:: std::string output_path

        Base directory path where all generated metric plots and reports will be saved. This path is typically derived from the model's checkpoint directory during registration.

    .. cpp:member:: const std::vector<Color_t> colors_h

        A predefined constant vector of ROOT color constants (``Color_t``). Used cyclically to assign distinct colors when plotting multiple datasets (e.g., training, validation, evaluation curves) or different variables on the same plot.

    .. cpp:member:: settings_t m_settings

        Structure holding various analysis settings loaded from configuration. This includes parameters like the total number of epochs, output paths, target variable names for mass reconstruction, physics parameters for mass calculation, etc.

    .. cpp:function:: void dump_plots(int k)

        Generates and saves all standard performance plots for a specified k-fold.

        This is a high-level convenience function that orchestrates the generation of
        multiple types of plots. It calls :cpp:func:`dump_loss_plots()` and :cpp:func:`dump_accuracy_plots()`.
        Additionally, if mass reconstruction targets are defined in ``m_settings.targets``,
        it also calls :cpp:func:`dump_mass_plots()`. Plots are saved to subdirectories within the
        ``output_path`` specific to the k-fold.

        :param k: The zero-based index of the k-fold for which to generate plots.

        .. seealso:: :cpp:func:`dump_loss_plots(int k)`, :cpp:func:`dump_accuracy_plots(int k)`, :cpp:func:`dump_mass_plots(int k)`

    .. cpp:function:: void dump_loss_plots(int k)

        Generates and saves plots visualizing the loss progression over epochs for a specific k-fold.

        This function focuses specifically on loss metrics. It retrieves the loss histograms
        (training, validation, evaluation) for each registered loss variable (graph, node, edge level)
        associated with the k-fold ``k``. It uses :cpp:func:`build_graphs` to convert these histograms into
        ``TGraph`` objects representing loss vs. epoch. Then, for each loss variable, it calls
        :cpp:func:`generic_painter` to create a plot showing the training, validation, and evaluation loss
        curves on a single canvas. The plots are saved into structured directories like
        ``output_path/k<k+1>/loss-graph/``, ``output_path/k<k+1>/loss-node/``, etc.

        :param k: The zero-based index of the k-fold for which to generate loss plots.

        .. seealso:: :cpp:func:`build_graphs()`, :cpp:func:`generic_painter()`

    .. cpp:function:: void dump_accuracy_plots(int k)

        Generates and saves plots visualizing the accuracy progression over epochs for a specific k-fold.

        Similar to :cpp:func:`dump_loss_plots`, but focuses on accuracy metrics. It retrieves accuracy
        histograms for the k-fold ``k``, converts them to ``TGraph`` objects using :cpp:func:`build_graphs`,
        and uses :cpp:func:`generic_painter` to plot the training, validation, and evaluation accuracy
        curves for each registered accuracy variable (graph, node, edge level). The Y-axis is
        labeled "MVA Accuracy (%)". Plots are saved into structured directories like
        ``output_path/k<k+1>/accuracy-graph/``, ``output_path/k<k+1>/accuracy-node/``, etc.

        :param k: The zero-based index of the k-fold for which to generate accuracy plots.

        .. seealso:: :cpp:func:`build_graphs()`, :cpp:func:`generic_painter()`

    .. cpp:function:: void dump_mass_plots(int k)

        Generates and saves invariant mass reconstruction plots comparing predictions to truth for a specific k-fold.

        This function creates detailed comparison plots for invariant mass distributions, specifically
        targeting variables defined in ``m_settings.targets``. For each target variable and the specified k-fold ``k``:
        1. Retrieves predicted mass histograms (``pred_mass_edge``) for training, validation, and evaluation modes.
        2. Retrieves the corresponding truth mass histograms (``truth_mass_edge``). Merges the statistics
            from the three modes into a single combined truth histogram for comparison.
        3. Creates a ``THStack`` to overlay the predicted histograms (train, valid, eval).
        4. Creates a ``TLegend`` to identify the different histograms (predictions and truth).
        5. Assigns colors/styles and adds histograms to the stack and legend.
        6. Creates a ``TCanvas`` and a ``TRatioPlot``. The ratio plot compares the stacked predictions
            against the combined truth histogram, showing the data/MC ratio (or prediction/truth ratio).
        7. Configures titles, draws the ratio plot and legend, sets the upper pad Y-axis to logarithmic scale.
        8. Saves the resulting canvas to a file path like ``output_path/k<k+1>/masses/<variable_name>/fold_<k+1>/epoch_<epoch+1>.png``.
        9. Cleans up allocated ROOT objects (TRatioPlot, THStack, merged truth histogram, legend, TCanvas).
        10. Resets the individual predicted mass histograms for potential reuse (e.g., if called multiple times).

        :param k: The zero-based index of the k-fold for which to generate mass plots.

        .. note:: This function assumes mass reconstruction is based on edge-level information.
        .. warning:: Requires ROOT libraries with TRatioPlot support.

    .. cpp:function:: model_report* register_model(model_template* model, int kfold)

        Registers a model instance with the metrics manager for a specific k-fold and initializes its analytics structures.

        This function sets up the necessary data structures within the ``registry`` to track metrics
        for the given model (``mod``) during the specified k-fold (``kfold``).
        1. Creates or accesses the ``analytics_t`` structure for ``kfold`` in the ``registry``.
        2. Stores the pointer ``mod`` in ``analytics_t::model``.
        3. Creates a new ``model_report`` object and stores its pointer in ``analytics_t::report``.
        4. Sets the base ``output_path`` for metrics based on the model's checkpoint path.
        5. Calls :cpp:func:`build_th1f_loss` for graph, node, and edge outputs defined in the model to initialize loss histograms.
        6. Calls :cpp:func:`build_th1f_accuracy` for graph, node, and edge outputs to initialize accuracy histograms.
        7. If ``m_settings.targets`` is not empty, calls :cpp:func:`build_th1f_mass` for each target variable to initialize
            both truth and predicted invariant mass histograms.

        :param model: Pointer to the ``model_template`` object being tracked.
        :param kfold: The zero-based index of the k-fold this model instance corresponds to.
        :return: A pointer to the newly created ``model_report`` structure associated with this k-fold's analytics.
                    This report structure will be populated with final metric values later.

        .. seealso:: :cpp:func:`build_th1f_loss()`, :cpp:func:`build_th1f_accuracy()`, :cpp:func:`build_th1f_mass()`, :cpp:struct:`analytics_t`, :cpp:struct:`model_report`

    .. cpp:function:: void capture(mode_enum mode, int kfold, int epoch, int smpl_len)

        Captures and records performance metrics after a processing step (e.g., batch or epoch).

        This function is called typically after a forward pass and loss calculation during training,
        validation, or evaluation. It retrieves the latest outputs and losses from the registered
        model for the specified ``kfold`` and updates the corresponding histograms for the given ``mode`` and ``epoch``.
        1. Retrieves the ``analytics_t`` structure for ``kfold`` and updates ``analytics_t::this_epoch``.
        2. Gets the calculated loss values (graph, node, edge) from the model (``model->m_p_loss``).
        3. Calls :cpp:func:`add_th1f_loss` for each loss type to fill the histograms, normalizing by ``smpl_len``.
        4. Iterates through the model's defined outputs (graph, node, edge):
            - Retrieves prediction tensors (``model->m_p_graph``, etc.) and truth tensors (``model->m_o_graph``, etc.).
            - Calls :cpp:func:`add_th1f_accuracy` to calculate accuracy and fill the corresponding histograms, normalizing by ``smpl_len``.
        5. If mass reconstruction targets (``m_settings.targets``) are defined:
            - Constructs the 4-momentum tensor (``pmc``) from node features specified in ``m_settings`` (pt, eta, phi, energy).
            - Transforms ``pmc`` to PxPyPzE format and scales units (e.g., MeV to GeV).
            - For each target variable:
              - Retrieves the relevant edge prediction and truth tensors.
              - Calls :cpp:func:`add_th1f_mass` to calculate invariant masses (predicted and true) and fill the respective histograms.
              - Includes checks for potentially invalid truth tensors.

        :param mode: The operational mode (``mode_enum::training``, ``mode_enum::validation``, ``mode_enum::evaluation``) for which metrics are being captured.
        :param kfold: The zero-based index of the k-fold being processed.
        :param epoch: The current epoch number (zero-based). This determines the bin to fill in the histograms.
        :param smpl_len: The number of samples or batches processed in this step. Used for averaging loss and accuracy values before filling histograms (e.g., if losses/accuracies are summed over a batch, ``smpl_len`` would be the batch size or number of batches).

        .. seealso:: :cpp:func:`add_th1f_loss()`, :cpp:func:`add_th1f_accuracy()`, :cpp:func:`add_th1f_mass()`, :cpp:struct:`analytics_t`

    **Private Members:**

    .. cpp:function:: void build_th1f_loss(std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>* type, graph_enum g_num, int kfold)

        Initializes TH1F histograms for tracking loss values over epochs for a specific graph element type.

        This function prepares the histograms needed to store loss values before training/evaluation begins.
        It iterates through the provided ``type`` map, which defines the loss variables associated with a
        specific graph element (``g_num``: graph, node, or edge). For each variable name found in the map:
        1. Creates three ``TH1F`` histograms: one for training loss, one for validation loss, and one for evaluation loss.
        2. Stores pointers to these histograms in the appropriate map within the ``analytics_t`` structure
            (``loss_graph``, ``loss_node``, or ``loss_edge``) for the specified ``kfold``, keyed by the ``mode_enum`` and variable name.
        3. Configures each histogram:
            - Sets the title (e.g., "Loss <VarName> <Mode> k<kfold+1>").
            - Sets the number of bins equal to the total number of epochs (``m_settings.epochs``).
            - Sets the X-axis range from 0 to ``m_settings.epochs``.
            - Sets X-axis title to "Epochs".
            - Sets Y-axis title to "Loss".

        :param type: Pointer to a map defining the loss outputs for the given element type. Keys are variable names (std::string).
                         Values are tuples ``std::tuple<torch::Tensor*, loss_enum>``, although the tensor pointer and enum within
                         the tuple appear unused in this specific function's logic, only the keys (variable names) are used.
        :param g_num: An enumeration value (``graph_enum``) indicating the graph element type these losses correspond to
                          (e.g., ``graph_enum::graph``, ``graph_enum::node``, ``graph_enum::edge``).
        :param kfold: The zero-based index of the k-fold for which histograms are being created.

        .. seealso:: :cpp:struct:`analytics_t`, :cpp:struct:`settings_t`

    .. cpp:function:: void add_th1f_loss(std::map<std::string, torch::Tensor>* type, std::map<std::string, TH1F*>* lss_type, int kfold, int smpl_len)

        Adds calculated loss values from PyTorch tensors to their corresponding TH1F histograms.

        This function takes a map of recently computed loss values (as tensors) and fills them
        into the appropriate pre-initialized histograms for the current epoch. It iterates through
        the input ``type`` map (mapping loss names to loss tensors). For each entry:
        1. Retrieves the scalar loss value from the tensor using ``.item<float>()``.
        2. Divides the loss value by ``smpl_len`` to get an average loss (assuming the input tensor
            contains a sum or average over ``smpl_len`` samples/batches).
        3. Finds the corresponding TH1F histogram in the ``lss_type`` map using the loss name.
        4. Fills the histogram bin corresponding to the current epoch (``registry[kfold].this_epoch + 1``)
            with the calculated average loss value. ROOT histograms are 1-indexed.

        :param type: Pointer to a map where keys are loss names (std::string) and values are ``torch::Tensor`` objects
                         containing the scalar loss values for the current step.
        :param[in,out] lss_type: Pointer to a map where keys are loss names and values are pointers to the ``TH1F``
                                    histograms that should be filled. These histograms are modified by this function.
        :param kfold: The zero-based index of the current k-fold. Used to access the current epoch number from the ``registry``.
        :param smpl_len: The number of samples or batches over which the loss in the ``type`` tensor was computed.
                              Used for normalization/averaging before filling the histogram.

        .. seealso:: :cpp:member:`analytics_t::this_epoch`

    .. cpp:function:: void build_th1f_accuracy(std::map<std::string, std::tuple<torch::Tensor*, loss_enum>>* type, graph_enum g_num, int kfold)

        Initializes TH1F histograms for tracking accuracy values over epochs for a specific graph element type.

        Similar to :cpp:func:`build_th1f_loss`, this function prepares histograms for storing accuracy percentages.
        It iterates through the ``type`` map (defining classification output variables for ``g_num``). For each variable:
        1. Creates three ``TH1F`` histograms (training, validation, evaluation accuracy).
        2. Stores pointers in the appropriate ``analytics_t`` map (``accuracy_graph``, ``accuracy_node``, or ``accuracy_edge``)
            for the given ``kfold``, keyed by ``mode_enum`` and variable name.
        3. Configures each histogram:
            - Sets the title (e.g., "Accuracy <VarName> <Mode> k<kfold+1>").
            - Sets the number of bins to ``m_settings.epochs``.
            - Sets the X-axis range from 0 to ``m_settings.epochs``.
            - Sets X-axis title to "Epochs".
            - Sets Y-axis title to "Accuracy (%)".
            - Sets the Y-axis range explicitly from 0 to 100.

        :param type: Pointer to a map defining the classification outputs. Keys are variable names (std::string).
                         Values are tuples ``std::tuple<torch::Tensor*, loss_enum>``, although the tensor pointer and enum within
                         the tuple appear unused in this specific function's logic, only the keys (variable names) are used.
        :param g_num: An enumeration value (``graph_enum``) indicating the graph element type this accuracy corresponds to
                          (e.g., ``graph_enum::graph``, ``graph_enum::node``, ``graph_enum::edge``).
        :param kfold: The zero-based index of the k-fold for which histograms are being created.

        .. seealso:: :cpp:struct:`analytics_t`, :cpp:struct:`settings_t`

    .. cpp:function:: void add_th1f_accuracy(torch::Tensor* pred, torch::Tensor* truth, TH1F* hist, int kfold, int smpl_len)

        Calculates classification accuracy and adds the value to a TH1F histogram.

        Computes the accuracy by comparing model predictions (``pred``) with ground truth labels (``truth``).
        1. Determines the predicted class for each sample by finding the index of the maximum value
            along the last dimension of the ``pred`` tensor (using ``torch::argmax``). Assumes ``pred`` contains
            logits or probabilities with shape like (..., num_classes).
        2. Compares the predicted class indices with the ``truth`` tensor containing the true class indices.
        3. Calculates the fraction of correct predictions (number correct / total samples).
        4. Scales the fraction by 100 to convert it to a percentage.
        5. Divides the percentage by ``smpl_len`` for averaging (consistent with how loss is handled).
        6. Finds the histogram bin corresponding to the current epoch (``registry[kfold].this_epoch + 1``).
        7. Fills the bin of the provided ``hist`` with the calculated average accuracy percentage.

        :param pred: Pointer to the raw prediction tensor from the model (e.g., logits or probabilities).
                         Shape is typically (batch_size, num_classes) or similar for node/edge predictions.
        :param truth: Pointer to the ground truth tensor containing the true class labels/indices.
                          Shape typically matches the prediction tensor's dimensions except for the class dimension (e.g., (batch_size)).
        :param[in,out] hist: Pointer to the ``TH1F`` histogram where the calculated accuracy percentage will be added.
                              The histogram is modified by this function.
        :param kfold: The zero-based index of the current k-fold. Used to access the current epoch number.
        :param smpl_len: The number of samples or batches over which the accuracy is calculated. Used for normalization/averaging.

        .. seealso:: :cpp:member:`analytics_t::this_epoch`

    .. cpp:function:: void build_th1f_mass(std::string var_name, graph_enum typ, int kfold)

        Initializes TH1F histograms for storing invariant mass distributions (truth and predicted).

        This function creates the necessary histograms for analyzing invariant mass reconstruction performance
        for a specific target particle or decay (``var_name``). It sets up histograms for both the distribution
        derived from ground truth information and the distribution derived from model predictions.
        1. Determines whether to create histograms for truth (``typ == graph_enum::truth_edge``) or predicted
            (``typ == graph_enum::data_edge``) masses based on the ``typ`` parameter.
        2. Creates three ``TH1F`` histograms (training, validation, evaluation) for the specified type (truth or predicted).
        3. Stores pointers to these histograms in the appropriate map within the ``analytics_t`` structure
            (``truth_mass_edge`` or ``pred_mass_edge``) for the given ``kfold``, keyed by ``mode_enum`` and ``var_name``.
        4. Configures each histogram:
            - Sets the title (e.g., "Mass <VarName> <Type> <Mode> k<kfold+1>").
            - Sets the number of bins and range based on ``m_settings.nbins`` and ``m_settings.max_range``.
            - Sets X-axis title to "Mass [GeV]" (assuming GeV units).
            - Sets Y-axis title to "Events".

        :param var_name: The name of the target variable (e.g., "TopQuark", "ZBoson") for which the invariant mass is being reconstructed.
        :param typ: An enumeration value (``graph_enum``) indicating the type of mass histogram to create:
                        ``graph_enum::truth_edge`` for histograms based on ground truth, or
                        ``graph_enum::data_edge`` (or similar) for histograms based on model predictions.
        :param kfold: The zero-based index of the k-fold for which histograms are being created.

        .. seealso:: :cpp:struct:`analytics_t`, :cpp:struct:`settings_t`

    .. cpp:function:: void add_th1f_mass(torch::Tensor* pmc, torch::Tensor* edge_index, torch::Tensor* truth, torch::Tensor* pred, int kfold, mode_enum mode, std::string var_name)

        Calculates predicted and true invariant masses from edge information and fills corresponding histograms.

        This function performs the core calculation for invariant mass reconstruction based on edge predictions
        and compares it to the mass calculated from ground truth edge information.
        1. **Predicted Mass:** Uses ``pyc::graph::edge_aggregation`` (presumably a function for message passing or aggregation)
            to sum the 4-momenta (``pmc``) of nodes connected by edges. The aggregation is weighted by the model's
            edge prediction probabilities (``pred``) for the target class (implicitly assumes class 1 represents the connection).
            Calculates the invariant mass (``M = sqrt(E^2 - Px^2 - Py^2 - Pz^2)``) of the resulting summed 4-momenta tensor.
        2. **True Mass:** Repeats the aggregation process, but uses the ground truth edge labels (``truth``) as weights.
            The sparse ``truth`` tensor (containing class indices) is likely converted to a one-hot format suitable for weighting
            before aggregation. Handles potential edge cases, like when only one node contributes (mass might be set to 0).
            Calculates the invariant mass from the truth-aggregated 4-momenta.
        3. **Filtering:** Removes entries with zero or negative calculated mass values from both predicted and true mass tensors.
        4. **Data Transfer:** Moves the resulting mass tensors (predicted and true) from the GPU (if applicable) to the CPU.
        5. **Synchronization:** Synchronizes CUDA streams if necessary to ensure calculations are complete before proceeding.
        6. **Conversion:** Converts the PyTorch mass tensors into ``std::vector<double>``.
        7. **Filling:** Iterates through the vectors and fills the appropriate histograms:
            - Fills the predicted mass histogram (``analytics_t::pred_mass_edge[mode][var_name]``) with the calculated predicted masses.
            - Fills the truth mass histogram (``analytics_t::truth_mass_edge[mode][var_name]``) with the calculated true masses.
            The histograms correspond to the specified ``kfold`` and operational ``mode``.

        :param pmc: Pointer to a tensor containing the 4-momenta (Px, Py, Pz, E) of the graph nodes. Assumed shape (num_nodes, 4). Units are expected to be consistent (e.g., GeV after potential scaling in :cpp:func:`capture`).
        :param edge_index: Pointer to the edge index tensor defining the graph connectivity. Shape typically (2, num_edges).
        :param truth: Pointer to the ground truth tensor for the edge classification task associated with ``var_name``. Shape typically (num_edges). Contains true class labels.
        :param pred: Pointer to the prediction tensor (e.g., probabilities or logits) for the edge classification task. Shape typically (num_edges, num_classes).
        :param kfold: The zero-based index of the current k-fold.
        :param mode: The current operational mode (``mode_enum::training``, ``mode_enum::validation``, ``mode_enum::evaluation``).
        :param var_name: The name of the target variable for which mass is being calculated (used as a key for histograms).

        .. seealso:: :cpp:struct:`analytics_t`, :cpp:func:`capture()`
        .. note:: The specific implementation of ``pyc::graph::edge_aggregation`` is external to this class but crucial for the calculation.

    .. cpp:function:: void generic_painter(std::vector<TGraph*> k_graphs, std::string path, std::string title, std::string xtitle, std::string ytitle, int epoch)

        Generic function to paint multiple TGraph objects onto a single TCanvas and save it.

        This utility function provides a standardized way to plot multiple ``TGraph`` objects,
        typically representing the evolution of a metric over epochs for different modes
        (e.g., training, validation, evaluation).
        1. Creates a ``TCanvas`` object.
        2. Creates a ``TMultiGraph`` object to hold the individual graphs.
        3. Iterates through the input ``k_graphs`` vector:
            - Sets the line width (e.g., to 2).
            - Assigns a color from the ``colors_h`` vector cyclically.
            - Adds the ``TGraph`` to the ``TMultiGraph``.
        4. Draws the ``TMultiGraph`` using the "APL" option (Axes, Points, Lines).
        5. Sets the main title of the plot using the ``title`` parameter.
        6. Sets the X-axis and Y-axis titles using ``xtitle`` and ``ytitle``.
        7. Configures the X-axis range from 0 to ``epoch`` (typically number of epochs).
        8. Adjusts axis label/title sizes and offsets for better readability.
        9. Builds a ``TLegend`` based on the titles of the individual ``TGraph`` objects in ``k_graphs``.
        10. Draws the legend.
        11. Saves the canvas to the specified ``path`` (e.g., as a PNG, PDF, etc.).
        12. Cleans up by deleting the allocated ``TCanvas`` and ``TMultiGraph`` objects.

        :param k_graphs: A ``std::vector`` containing pointers to the ``TGraph`` objects to be plotted together. Each graph's title is used for the legend entry.
        :param path: The full file path (including filename and extension, e.g., "/path/to/plot.png") where the generated plot image will be saved.
        :param title: The main title string to be displayed at the top of the plot.
        :param xtitle: The title string for the X-axis (typically "Epochs").
        :param ytitle: The title string for the Y-axis (e.g., "Loss", "Accuracy (%)").
        :param epoch: The maximum value for the X-axis range. This should typically correspond to the total number of epochs the graphs represent.

        .. seealso:: :cpp:func:`dump_loss_plots()`, :cpp:func:`dump_accuracy_plots()`, :cpp:member:`colors_h`

    .. cpp:function:: std::map<std::string, std::vector<TGraph*>> build_graphs(std::map<std::string, TH1F*>* train, std::map<std::string, float>* tr_, std::map<std::string, TH1F*>* valid, std::map<std::string, float>* va_, std::map<std::string, TH1F*>* eval, std::map<std::string, float>* ev_, int ep)

        Builds TGraph objects from TH1F histograms representing metric evolution and extracts final values.

        This function transforms the per-epoch data stored in TH1F histograms (where each bin represents an epoch)
        into TGraph objects suitable for plotting trends. It processes histograms for training, validation, and
        evaluation modes simultaneously for a given metric.
        1. Iterates through the keys (variable names) present in the ``train`` histogram map.
        2. For each variable name:
            - Creates three new ``TGraph`` objects: one for training, one for validation, and one for evaluation data.
            - Sets titles for the graphs (e.g., "Training", "Validation", "Evaluation") to be used in legends.
            - Iterates from epoch 0 to ``ep`` (exclusive):
              - Retrieves the metric value (bin content) from the corresponding training, validation, and evaluation TH1F histograms for that epoch (bin ``i+1``).
              - Adds a point ``(i, value)`` to the respective ``TGraph``.
            - Extracts the final metric value (at epoch ``ep``) from each histogram (bin ``ep+1``) and stores it
              in the corresponding report maps (``tr_``, ``va_``, ``ev_``), keyed by the variable name.
            - Stores the pointers to the three created ``TGraph`` objects (train, valid, eval) in a ``std::vector``.
            - Adds an entry to the output map, mapping the variable name to this vector of ``TGraph`` pointers.
        3. Returns the map containing the generated ``TGraph`` vectors, ready for plotting with :cpp:func:`generic_painter`.

        :param train: Pointer to a map storing training TH1F histograms (key: metric name, value: TH1F*). Each bin content represents the metric value at that epoch.
        :param[out] tr_: Pointer to a map where the final training metric value (at epoch ``ep``) will be stored (key: metric name, value: float).
        :param valid: Pointer to a map storing validation TH1F histograms.
        :param[out] va_: Pointer to a map where the final validation metric value will be stored.
        :param eval: Pointer to a map storing evaluation TH1F histograms.
        :param[out] ev_: Pointer to a map where the final evaluation metric value will be stored.
        :param ep: The epoch number whose value should be extracted as the "final" value for the report maps (``tr_``, ``va_``, ``ev_``). This typically corresponds to the last completed epoch or the epoch with the best validation performance. Note that graphs are filled up to epoch ``ep-1``.
        :return: A map (``std::map<std::string, std::vector<TGraph*>>``) where keys are metric names and values are vectors
                    containing three ``TGraph`` pointers: ``[training_graph, validation_graph, evaluation_graph]``.

        .. note:: The caller is responsible for deleting the ``TGraph`` objects created by this function after they are no longer needed (e.g., after plotting). :cpp:func:`generic_painter` does *not* delete the input graphs.
        .. seealso:: :cpp:func:`generic_painter()`, :cpp:func:`dump_loss_plots()`, :cpp:func:`dump_accuracy_plots()`

    .. cpp:member:: std::map<int, analytics_t> registry = {}

        The central registry storing all analytics data. It's a map where the key is the integer k-fold index (zero-based), and the value is the ``analytics_t`` structure containing all histograms, model pointers, and report pointers for that specific fold.


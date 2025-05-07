.. cpp:file:: plotting.h

    Defines the plotting class and related structures for generating plots using various backends.

    This header file declares the ``plotting`` class, which serves as a central hub for
    creating diverse plot types, such as histograms (1D and 2D) and Receiver Operating
    Characteristic (ROC) curves. It inherits utility functions from the ``tools`` class
    and notification capabilities from the ``notification`` class. The ``plotting`` class
    manages plot data (x/y coordinates, errors, weights), styling parameters (colors,
    line styles, markers, fonts), axis configurations (limits, labels, scaling, binning),
    and output file generation (path, filename, format). It also defines the ``roc_t``
    structure, a container specifically designed to hold the necessary data points
    (truth labels and prediction scores) for constructing a single ROC curve.

.. cpp:struct:: roc_t

    Structure to hold all necessary data for generating a single Receiver Operating Characteristic (ROC) curve.

    This structure encapsulates the ground truth labels and corresponding prediction scores
    required to compute and plot an ROC curve. It typically represents the performance
    of a specific machine learning model for a particular class, often within a specific
    cross-validation fold (k-fold).

    .. cpp:member:: int cls = 0

        The index of the class for which this ROC curve is generated.
        For binary classification, this might be 0 or 1. For multi-class
        problems, it indicates the specific class being treated as positive.
        Default value is 0.

    .. cpp:member:: int kfold = 0

        The index of the k-fold cross-validation split this data belongs to.
        If cross-validation is not used, this might be set to a default value like 0.
        Default value is 0.

    .. cpp:member:: std::string model = ""

        A string identifier for the model whose performance is represented by this ROC curve.
        This helps distinguish between different models when plotting multiple curves.
        Default is an empty string.

    .. cpp:member:: std::vector<std::vector<int>>* truth = nullptr

        Pointer to a vector of vectors containing the ground truth labels in one-hot encoded format.
        Each inner vector corresponds to a single data sample. The inner vector has a size
        equal to the number of classes, with a '1' at the index corresponding to the true
        class and '0's elsewhere. The ``plotting::build_ROC`` method handles the conversion
        from simple integer labels to this format if needed.
        Default is nullptr. The memory pointed to is managed by the ``plotting`` class.

    .. cpp:member:: std::vector<std::vector<double>>* scores = nullptr

        Pointer to a vector of vectors containing the prediction scores or probabilities from the model.
        Each inner vector corresponds to a single data sample and contains the scores
        assigned by the model to each class for that sample. The order of scores should
        match the order of classes in the ``truth`` labels.
        Default is nullptr. The memory pointed to is managed by the ``plotting`` class.

.. cpp:class:: plotting : public tools, public notification

    Provides a comprehensive interface for configuring, managing data for, and generating various plots.

    This class acts as a high-level manager for creating scientific plots. It allows users
    to define plot appearance (titles, labels, colors, styles, fonts), configure axes
    (ranges, logarithmic scales, binning), provide data (x/y values, errors, weights),
    and specify output details (path, filename, format, resolution). It includes specific
    functionality for handling data required for ROC curve generation (``build_ROC``, ``get_ROC``).
    It leverages functionalities inherited from the ``tools`` base class (e.g., file system
    operations, data manipulation) and the ``notification`` base class (e.g., logging messages).
    The actual plotting is typically delegated to a backend implementation (not defined here).

    **Public Members**

    .. cpp:function:: plotting()

        Default constructor for the plotting class.
        Initializes a ``plotting`` object with default values for all its member variables,
        setting up a basic plot configuration (e.g., output path "./Figures", filename "untitled",
        extension ".pdf", default titles/labels, standard linear axes, etc.).

    .. cpp:function:: ~plotting()

        Destructor for the plotting class.
        Responsible for cleaning up dynamically allocated memory. Specifically, it iterates
        through the ``roc_data`` and ``labels`` maps and deletes the allocated vectors
        storing the ROC curve scores and truth labels to prevent memory leaks.

    .. cpp:function:: std::string build_path()

        Constructs the full output file path, ensuring the directory structure exists.

        This method combines the ``output_path``, ``filename``, and ``extension`` members to form
        the complete path for the plot output file. It checks if the ``output_path`` directory
        exists and creates it if it doesn't. It also ensures a subdirectory named 'raw/'
        exists within the ``output_path``, creating it if necessary. This 'raw/' directory
        might be used by backend implementations to store intermediate data.

        :returns: The fully qualified path (including directory, filename, and extension) where the plot output file should be saved.

    .. cpp:function:: float get_max(std::string dim)

        Retrieves the maximum value from the specified data vector (``x_data`` or ``y_data``).

        :param dim: A string indicating the dimension: "x" for ``x_data``, "y" for ``y_data``.
        :returns: The maximum value found in the specified data vector. Returns 1.0f if the dimension string is invalid or if the corresponding data vector is empty. Uses the ``max`` utility function inherited from the ``tools`` base class.

    .. cpp:function:: float get_min(std::string dim)

        Retrieves the minimum value from the specified data vector (``x_data`` or ``y_data``).

        :param dim: A string indicating the dimension: "x" for ``x_data``, "y" for ``y_data``.
        :returns: The minimum value found in the specified data vector. Returns 1.0f if the dimension string is invalid or if the corresponding data vector is empty. Uses the ``min`` utility function inherited from the ``tools`` base class.

    .. cpp:function:: float sum_of_weights()

        Calculates the sum of all elements in the ``weights`` vector.

        This is often used for normalizing weighted histograms or calculating effective
        numbers of entries.

        :returns: The total sum of the weights. Returns 1.0f if the ``weights`` vector is empty or if the calculated sum is zero to avoid division-by-zero errors in subsequent calculations.

    .. cpp:function:: void build_error()

        Calculates mean and standard deviation for y-values grouped by unique x-values, populating error vectors.

        This method processes potentially redundant data points where multiple ``y_data`` values
        might share the same ``x_data`` value. It groups ``y_data`` values based on their corresponding
        unique ``x_data`` value. For each group, it calculates the mean and standard deviation
        of the y-values. The original ``x_data`` and ``y_data`` vectors are then cleared and
        repopulated with the unique x-values and their corresponding mean y-values.
        The ``y_error_up`` and ``y_error_down`` vectors are populated with the mean +/- standard deviation,
        respectively. If ``y_error_down`` is already populated (indicating errors have been calculated
        or provided), this function does nothing to avoid overwriting existing error data.
        Negative lower error bounds are clamped to zero.

    .. cpp:function:: std::tuple<float, float> mean_stdev(std::vector<float>* data)

        Calculates the sample mean and sample standard deviation of a given vector of floats.

        :param data: A pointer to a ``std::vector<float>`` containing the dataset.
        :returns: A tuple where the first element is the calculated mean and the second element is the calculated sample standard deviation (using N-1 denominator). If the input vector contains fewer than 2 elements, it returns the first element (or 0 if empty, though the calling context ``build_error`` ensures at least one element) and a standard deviation of 0.

    .. cpp:function:: void build_ROC(std::string name, int kfold, std::vector<int>* labels, std::vector<std::vector<double>>* scores)

        Stores ROC curve data (truth labels and prediction scores) internally for later retrieval.

        This function takes ground truth labels and model prediction scores, associates them
        with a given model name and k-fold index, and stores them in the internal ``labels``
        and ``roc_data`` maps. It handles dynamic memory allocation for storing copies of the
        score and label data. If data for the specified ``name`` and ``kfold`` already exists,
        it is deleted and replaced. The input ``labels`` (vector of ints) are converted into
        a one-hot encoded format (vector of vectors of ints) before storing in the ``labels`` map.
        The number of classes is inferred from the data.

        :param name: The identifier (e.g., model name) for this set of ROC data.
        :param kfold: The cross-validation fold index associated with this data.
        :param labels: Pointer to a vector of integer ground truth class labels for each sample. The function will convert this to one-hot encoding internally.
        :param scores: Pointer to a vector of vectors, where each inner vector contains the prediction scores (e.g., probabilities) for each class for a given sample. The outer vector size should match ``labels->size()``.

    .. cpp:function:: std::vector<roc_t> get_ROC()

        Retrieves all stored ROC data, packaged into ``roc_t`` structures.

        This method iterates through the internal ``roc_data`` and ``labels`` maps, which store
        the scores and truth labels organized by model name and k-fold index. For each
        unique combination of model and k-fold, it creates a ``roc_t`` object, populates it
        with the corresponding data (pointers to the internally managed score and label vectors,
        model name, k-fold index, and number of classes), and adds it to a result vector.

        :returns: A vector where each element is a ``roc_t`` structure containing the necessary data to generate one ROC curve. The pointers within the ``roc_t`` structures point to memory managed by the ``plotting`` object.

    **IO Members**

    .. cpp:member:: std::string extension = ".pdf"

        The file extension determining the output format of the plot (e.g., ".pdf", ".png", ".svg"). Default is ".pdf".

    .. cpp:member:: std::string filename = "untitled"

        The base name for the output plot file (without the directory path or extension). Default is "untitled".

    .. cpp:member:: std::string output_path = "./Figures"

        The path to the directory where the output plot files will be saved. The ``build_path`` method will ensure this directory exists. Default is "./Figures".

    **Meta Data Members**

    .. cpp:member:: float x_min = 0

        The explicit minimum value for the x-axis. If ``auto_scale`` is true, this might be overridden. Default is 0.

    .. cpp:member:: float y_min = 0

        The explicit minimum value for the y-axis. If ``auto_scale`` is true, this might be overridden. Default is 0.

    .. cpp:member:: float x_max = 0

        The explicit maximum value for the x-axis. If ``auto_scale`` is true, this might be overridden. Default is 0.

    .. cpp:member:: float y_max = 0

        The explicit maximum value for the y-axis. If ``auto_scale`` is true, this might be overridden. Default is 0.

    .. cpp:member:: int x_bins = 100

        The number of bins to use for the x-axis when generating a histogram. This is ignored if ``variable_x_bins`` is non-empty. Default is 100.

    .. cpp:member:: int y_bins = 100

        The number of bins to use for the y-axis when generating a 2D histogram. This is ignored if ``variable_y_bins`` is non-empty. Default is 100.

    .. cpp:member:: bool errors = false

        Flag to control whether error bars (defined by ``y_error_up``, ``y_error_down``) should be drawn on the plot. The ``build_error`` method can be used to calculate errors if they are not provided directly. Default is false.

    .. cpp:member:: bool counts = false

        Flag to control whether the count (bin content) should be displayed as text on top of histogram bins. Applicable primarily to 1D histograms. Default is false.

    **Cosmetics Members**

    .. cpp:member:: std::string style = "ROOT"

        Specifies a plotting style preset (e.g., "ROOT", "matplotlib", "seaborn"). The interpretation of this string depends on the plotting backend implementation. Default is "ROOT".

    .. cpp:member:: std::string title = "untitled"

        The main title displayed at the top of the plot. Default is "untitled".

    .. cpp:member:: std::string ytitle = "y-axis"

        The label displayed along the y-axis. Default is "y-axis".

    .. cpp:member:: std::string xtitle = "x-axis"

        The label displayed along the x-axis. Default is "x-axis".

    .. cpp:member:: std::string histfill = "fill"

        Specifies the drawing style for histograms (e.g., "fill", "step", "bar"). "fill" usually means filled bars, "step" means outlined bars. Interpretation depends on the backend. Default is "fill".

    .. cpp:member:: std::string overflow = "sum"

        Specifies how to handle histogram entries that fall outside the defined axis range (overflow/underflow). Common options might include "sum" (add to edge bins) or "ignore". Interpretation depends on the backend. Default is "sum".

    .. cpp:member:: std::string marker = "."

        The style of markers used for scatter plots or data points on line plots (e.g., ".", "o", "x", "+", "*"). Interpretation depends on the backend. Default is ".".

    .. cpp:member:: std::string hatch = ""

        A pattern used to fill shapes (e.g., histogram bars, areas under curves) like "/", "\\", "|", "-". Useful for distinguishing stacked histograms or adding texture. Default is "" (no hatch).

    .. cpp:member:: std::string linestyle = "-"

        The style of lines used in line plots (e.g., "-", "--", ":", "-."). Solid, dashed, dotted, dash-dot, etc. Interpretation depends on the backend. Default is "-".

    .. cpp:member:: std::string color = ""

        A specific color to use for plot elements (lines, markers, fills). If this is empty, the plotting backend will typically cycle through the colors defined in the ``colors`` vector. Color names (e.g., "red", "blue") or hex codes might be supported depending on the backend. Default is "" (use ``colors`` vector or backend default).

    .. cpp:member:: std::vector<std::string> colors = {}

        A list of color identifiers (names, hex codes) to be used cyclically when plotting multiple datasets (e.g., multiple histograms on the same axes, multiple lines). Default is an empty vector (backend chooses default colors).

    .. cpp:member:: bool stack = false

        Flag indicating whether multiple histograms plotted on the same axes should be stacked on top of each other. If false, they are typically overlaid. Default is false.

    .. cpp:member:: bool density = false

        Flag indicating whether histograms should be normalized to represent a probability density (i.e., the total area under the histogram equals 1). If false, histograms show raw counts. Default is false.

    .. cpp:member:: bool x_logarithmic = false

        Flag indicating whether the x-axis should use a logarithmic scale. Default is false (linear scale).

    .. cpp:member:: bool y_logarithmic = false

        Flag indicating whether the y-axis should use a logarithmic scale. Useful for visualizing data spanning multiple orders of magnitude. Default is false (linear scale).

    .. cpp:member:: float line_width = 0.1

        The width of lines drawn on the plot (e.g., line plots, histogram outlines). Units depend on the backend. Default is 0.1.

    .. cpp:member:: float cap_size = 1.0

        The size of the caps drawn at the ends of error bars. Units depend on the backend. Default is 1.0.

    .. cpp:member:: float alpha = 0.4

        The alpha transparency level for plot elements (fills, lines, markers). Value ranges from 0.0 (completely transparent) to 1.0 (completely opaque). Useful for visualizing overlapping data. Default is 0.4.

    .. cpp:member:: float x_step = -1

        The desired step size between major ticks on the x-axis. If set to -1, the plotting backend will determine the tick spacing automatically. Default is -1 (automatic).

    .. cpp:member:: float y_step = -1

        The desired step size between major ticks on the y-axis. If set to -1, the plotting backend will determine the tick spacing automatically. Default is -1 (automatic).

    **Fonts Members**

    .. cpp:member:: float font_size = 10

        The general font size for text elements like annotations (unless overridden). Units depend on the backend (e.g., points). Default is 10.

    .. cpp:member:: float axis_size = 12.5

        The font size specifically for the axis labels (xtitle, ytitle) and tick labels. Units depend on the backend. Default is 12.5.

    .. cpp:member:: float legend_size = 10

        The font size for the text within the plot legend. Units depend on the backend. Default is 10.

    .. cpp:member:: float title_size = 10

        The font size for the main plot title. Units depend on the backend. Default is 10.

    .. cpp:member:: bool use_latex = true

        Flag indicating whether LaTeX should be used for rendering text elements (titles, labels, annotations). Requires a backend that supports LaTeX rendering (e.g., matplotlib with a LaTeX installation). Allows for complex mathematical formulas in text. Default is true.

    **Scaling Members**

    .. cpp:member:: int dpi = 400

        The resolution (Dots Per Inch) for raster image output formats (like PNG, JPG). Higher values result in higher resolution images. Default is 400.

    .. cpp:member:: float xscaling = 1.25*6.4

        A scaling factor applied to the default horizontal size of the plot figure. Allows adjustment of the plot's aspect ratio and overall size. Default is 1.25 * 6.4 (based on typical matplotlib default width).

    .. cpp:member:: float yscaling = 1.25*4.8

        A scaling factor applied to the default vertical size of the plot figure. Allows adjustment of the plot's aspect ratio and overall size. Default is 1.25 * 4.8 (based on typical matplotlib default height).

    .. cpp:member:: bool auto_scale = true

        Flag indicating whether the plot axes should automatically scale to fit the range of the data. If true, ``x_min``, ``x_max``, ``y_min``, ``y_max`` might be ignored or used as initial hints. If false, the explicit limits (``x_min``, etc.) are strictly enforced. Default is true.

    **Data Containers**

    .. cpp:member:: std::vector<float> x_data = {}

        Vector storing the primary x-coordinates for plotting (e.g., scatter plot x-values, histogram data).

    .. cpp:member:: std::vector<float> y_data = {}

        Vector storing the primary y-coordinates for plotting (e.g., scatter plot y-values, 2D histogram y-values). Should typically have the same size as ``x_data`` for point-based plots.

    .. cpp:member:: std::map<std::string, std::map<int, std::vector<std::vector<double>>*>> roc_data = {}

        Nested map storing prediction scores for ROC curve generation.
        Outer map key: Model name (std::string).
        Inner map key: K-fold index (int).
        Value: Pointer to a ``std::vector<std::vector<double>>`` containing the scores.
        Memory is managed by ``build_ROC`` and the destructor.

    .. cpp:member:: std::map<std::string, std::map<int, std::vector<std::vector<int>>*>> labels = {}

        Nested map storing ground truth labels (one-hot encoded) for ROC curve generation.
        Outer map key: Model name (std::string).
        Inner map key: K-fold index (int).
        Value: Pointer to a ``std::vector<std::vector<int>>`` containing the labels.
        Memory is managed by ``build_ROC`` and the destructor.

    .. cpp:member:: std::vector<float> y_error_up = {}

        Vector storing the upper bounds for y-axis error bars. Should have the same size as the plotted ``y_data``. Can be populated manually or by the ``build_error`` method.

    .. cpp:member:: std::vector<float> y_error_down = {}

        Vector storing the lower bounds for y-axis error bars. Should have the same size as the plotted ``y_data``. Can be populated manually or by the ``build_error`` method.

    .. cpp:member:: std::unordered_map<std::string, float> x_labels = {}

        Map associating string labels with specific numerical positions on the x-axis. Used for creating categorical axes where ticks are labeled with text instead of numbers. Key: String label. Value: Float position on the axis.

    .. cpp:member:: std::unordered_map<std::string, float> y_labels = {}

        Map associating string labels with specific numerical positions on the y-axis. Used for creating categorical axes. Key: String label. Value: Float position on the axis.

    .. cpp:member:: std::vector<float> variable_x_bins = {}

        Vector defining custom bin edges for the x-axis histogram. If this vector is non-empty, it overrides the ``x_bins`` setting, allowing for variable-width bins. The values should be monotonically increasing.

    .. cpp:member:: std::vector<float> variable_y_bins = {}

        Vector defining custom bin edges for the y-axis histogram (for 2D plots). If non-empty, it overrides ``y_bins``, allowing variable-width bins on the y-axis. Values should be monotonically increasing.

    .. cpp:member:: std::vector<float> weights = {}

        Vector storing weights associated with each data point (``x_data``, ``y_data``). Used for weighted histograms or other weighted calculations. If used, it should typically have the same size as ``x_data``. If empty, all points have a weight of 1.

    .. cpp:member:: float cross_section = -1

        A physics cross-section value, potentially used for scaling simulation data to match experimental data. Units are context-dependent (e.g., pb, fb). Default is -1 (indicating it's likely unused).

    .. cpp:member:: float integrated_luminosity = 140.1

        An integrated luminosity value, typically used in conjunction with ``cross_section`` for scaling simulation data. Commonly expressed in inverse femtobarns (fb^-1). Default is 140.1 fb^-1.

    **Private Members**

    .. cpp:function:: template <typename g> std::vector<std::vector<g>>* generate(size_t x, size_t y)

        Template function to dynamically allocate and initialize a 2D vector (vector of vectors) with zeros.

        This utility function creates a new 2D vector on the heap with the specified dimensions.
        All elements are initialized to the default value of type ``g`` (which is 0 for numeric types).
        This is primarily used internally by ``build_ROC`` to allocate storage for scores and labels.

        :tparam g: The data type of the elements within the inner vectors (e.g., ``int``, ``double``).
        :param x: The number of rows (size of the outer vector).
        :param y: The number of columns (size of each inner vector).
        :returns: A pointer to the newly allocated 2D vector. The caller (specifically, the ``plotting`` class destructor or ``build_ROC`` when replacing data) is responsible for ``delete``ing this pointer to prevent memory leaks.


Plotting Module
===============

The ``plotting`` class is the base for all visualisation helpers in AnalysisG.
It inherits from ``tools`` and ``notification`` and provides a rich set of
configurable fields for controlling histogram binning, colour, font, legend,
and axis parameters.  The derived ``roc`` class (see :doc:`roc`) uses this base.

Class: ``plotting``
--------------------

**Header:** ``<plotting/plotting.h>``

**Inheritance:** ``tools``, ``notification``

I/O Fields
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``extension``
     - ``std::string``
     - Output file extension.  Default ``".pdf"``.
   * - ``filename``
     - ``std::string``
     - Base output filename (without extension).  Default ``"untitled"``.
   * - ``output_path``
     - ``std::string``
     - Directory for saved figures.  Default ``"./Figures"``.

Binning / Range Fields
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``x_min``
     - ``float``
     - Lower bound of the x-axis.  Default ``0``.
   * - ``y_min``
     - ``float``
     - Lower bound of the y-axis.  Default ``0``.
   * - ``x_max``
     - ``float``
     - Upper bound of the x-axis.  Default ``0`` (auto-set if 0).
   * - ``y_max``
     - ``float``
     - Upper bound of the y-axis.  Default ``0`` (auto-set if 0).
   * - ``x_bins``
     - ``int``
     - Number of x-axis bins.  Default ``100``.
   * - ``y_bins``
     - ``int``
     - Number of y-axis bins.  Default ``100``.
   * - ``variable_x_bins``
     - ``std::vector<float>``
     - Variable-width x-bin edges (overrides ``x_bins`` when non-empty).
   * - ``variable_y_bins``
     - ``std::vector<float>``
     - Variable-width y-bin edges.
   * - ``errors``
     - ``bool``
     - Draw error bars.  Default ``false``.
   * - ``counts``
     - ``bool``
     - Use raw count instead of normalised density.  Default ``false``.

Cosmetic / Style Fields
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``style``
     - ``std::string``
     - Plot style.  Default ``"ROOT"``.
   * - ``title``
     - ``std::string``
     - Histogram title.  Default ``"untitled"``.
   * - ``ytitle``
     - ``std::string``
     - y-axis label.  Default ``"y-axis"``.
   * - ``xtitle``
     - ``std::string``
     - x-axis label.  Default ``"x-axis"``.
   * - ``histfill``
     - ``std::string``
     - Fill style (``"fill"``, ``"step"``, …).  Default ``"fill"``.
   * - ``overflow``
     - ``std::string``
     - Overflow handling (``"sum"``, ``"include"``, ``"ignore"``).  Default ``"sum"``.
   * - ``marker``
     - ``std::string``
     - Matplotlib marker style.  Default ``"."``.
   * - ``hatch``
     - ``std::string``
     - Hatch pattern for filled histograms.  Default ``""``.
   * - ``linestyle``
     - ``std::string``
     - Matplotlib line style.  Default ``"-"``.
   * - ``color``
     - ``std::string``
     - Single colour name/hex.  Default ``""`` (auto).
   * - ``colors``
     - ``std::vector<std::string>``
     - Per-histogram colour list.  Default ``{}``.
   * - ``stack``
     - ``bool``
     - Stack histograms.  Default ``false``.
   * - ``density``
     - ``bool``
     - Normalise histogram to unit area.  Default ``false``.
   * - ``x_logarithmic``
     - ``bool``
     - Logarithmic x-axis scale.  Default ``false``.
   * - ``y_logarithmic``
     - ``bool``
     - Logarithmic y-axis scale.  Default ``false``.
   * - ``line_width``
     - ``float``
     - Line width in points.  Default ``0.1``.
   * - ``cap_size``
     - ``float``
     - Error-bar cap size.  Default ``1.0``.
   * - ``alpha``
     - ``float``
     - Fill transparency (0–1).  Default ``0.4``.
   * - ``x_step``
     - ``float``
     - x-axis tick step (``-1`` = auto).  Default ``-1``.
   * - ``y_step``
     - ``float``
     - y-axis tick step (``-1`` = auto).  Default ``-1``.

Font Fields
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``font_size``
     - ``float``
     - General font size in points.  Default ``10``.
   * - ``axis_size``
     - ``float``
     - Axis label font size.  Default ``12.5``.
   * - ``legend_size``
     - ``float``
     - Legend font size.  Default ``10``.
   * - ``title_size``
     - ``float``
     - Title font size.  Default ``10``.
   * - ``use_latex``
     - ``bool``
     - Render labels using LaTeX.  Default ``true``.

Scaling Fields
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``dpi``
     - ``int``
     - Output image resolution (dots per inch).  Default ``400``.
   * - ``xscaling``
     - ``float``
     - Figure width scaling factor.  Default ``1.25 × 6.4``.
   * - ``yscaling``
     - ``float``
     - Figure height scaling factor.  Default ``1.25 × 4.8``.
   * - ``auto_scale``
     - ``bool``
     - Automatically scale figure dimensions.  Default ``true``.

Data / Weight Fields
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Field
     - Type
     - Description
   * - ``x_data``
     - ``std::vector<float>``
     - x-axis data points.
   * - ``y_data``
     - ``std::vector<float>``
     - y-axis data points.
   * - ``y_error_up``
     - ``std::vector<float>``
     - Positive error on each y point.
   * - ``y_error_down``
     - ``std::vector<float>``
     - Negative error on each y point.
   * - ``x_labels``
     - ``std::unordered_map<std::string, float>``
     - Custom x-axis tick labels.
   * - ``y_labels``
     - ``std::unordered_map<std::string, float>``
     - Custom y-axis tick labels.
   * - ``weights``
     - ``std::vector<float>``
     - Per-event histogram weights.
   * - ``cross_section``
     - ``float``
     - Dataset cross-section in fb.  Default ``-1`` (unused if negative).
   * - ``integrated_luminosity``
     - ``float``
     - Integrated luminosity in fb⁻¹.  Default ``140.1``.

Public Methods
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Signature
     - Description
   * - ``std::string build_path()``
     - Constructs the full output file path ``output_path/filename.extension``.
   * - ``float get_max(std::string dim)``
     - Returns the maximum value in the ``x_data`` (``dim="x"``) or ``y_data``
       (``dim="y"``) vector.
   * - ``float get_min(std::string dim)``
     - Returns the minimum value in the ``x_data`` or ``y_data`` vector.
   * - ``float sum_of_weights()``
     - Returns the sum of the ``weights`` vector.
   * - ``void build_error()``
     - Computes Poisson error bars from ``y_data`` and fills ``y_error_up`` and
       ``y_error_down``.
   * - ``std::tuple<float,float> mean_stdev(std::vector<float>* data)``
     - Returns the arithmetic mean and standard deviation of *data*.

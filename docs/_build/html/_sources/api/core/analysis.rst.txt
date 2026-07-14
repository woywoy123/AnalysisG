Analysis (Python)
=================

The ``Analysis`` Cython class is the top-level framework compiler exposed
to Python.  It owns the underlying C++ ``analysis`` pointer and keeps
Python-side references to every registered template so they are not
garbage-collected while C++ holds raw pointers to them.

Registration Methods
--------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method signature
     - Description
   * - ``AddSamples(path: str, label: str)``
     - Register a ROOT file or directory together with a sample label used
       for bookkeeping and event/graph matching.
   * - ``AddEvent(ev: EventTemplate, label: str)``
     - Register an :class:`EventTemplate` subclass instance associated with
       a sample label.
   * - ``AddGraph(g: GraphTemplate, label: str)``
     - Register a :class:`GraphTemplate` subclass instance associated with a
       sample label.
   * - ``AddSelection(s: SelectionTemplate)``
     - Register a :class:`SelectionTemplate` subclass for cut-based event
       selection.
   * - ``AddMetric(m: MetricTemplate, model: ModelTemplate)``
     - Register a :class:`MetricTemplate` paired with a model; also sets
       ``PreTagEvents = True`` internally.
   * - ``AddModel(model: ModelTemplate, op: OptimizerConfig, run_name: str)``
     - Register a model for training with an optimizer configuration and a
       run name string.
   * - ``AddModelInference(model: ModelTemplate, run_name: str = "run_name")``
     - Register a model for inference only (no optimizer configured).

Execution
---------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Method
     - Description
   * - ``Start()``
     - Launch the full analysis pipeline: meta-data fetching, graph caching,
       training, validation, evaluation, and selection post-processing.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Property
     - Default
     - Description
   * - ``GetMetaData`` *(read-only)*
     - —
     - Returns a :class:`MetaLookup` built from cached :class:`Meta` objects
       collected during ``Start()``.
   * - ``PreTagEvents``
     - —
     - Set automatically by ``AddMetric``; pre-tags events with model
       inference before running selections.

I/O Settings
------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Property
     - Default
     - Description
   * - ``OutputPath``
     - ``'./ProjectName'``
     - Directory where all results (graphs, plots, models) are written.
   * - ``GraphCache``
     - ``''``
     - Path to the graph cache HDF5 file.
   * - ``BuildCache``
     - ``False``
     - If ``True``, build (or rebuild) the graph HDF5 cache.
   * - ``TrainingDataset``
     - ``''``
     - Path to an existing HDF5 training dataset (``'.h5'`` appended
       automatically if missing).
   * - ``SumOfWeightsTreeName``
     - ``''``
     - Name of the ROOT tree containing sum-of-weights information.
   * - ``FetchMeta``
     - ``False``
     - If ``True``, fetch ATLAS dataset metadata from AMI during ``Start()``.
   * - ``SaveSelectionToROOT``
     - ``False``
     - Serialise selection results to a ROOT file alongside pickle output.
   * - ``VarPt``
     - ``b'pt'``
     - Branch name (bytes) mapped to transverse momentum.
   * - ``VarEta``
     - ``b'eta'``
     - Branch name (bytes) mapped to pseudorapidity.
   * - ``VarPhi``
     - ``b'phi'``
     - Branch name (bytes) mapped to azimuthal angle.
   * - ``VarEnergy``
     - ``b'energy'``
     - Branch name (bytes) mapped to energy.

ML Settings
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Property
     - Default
     - Description
   * - ``Epochs``
     - ``10``
     - Number of training epochs.
   * - ``kFolds``
     - ``10``
     - Total number of cross-validation folds.
   * - ``kFold``
     - ``[]``
     - List of fold indices to actually run; ``[]`` means run all folds.
   * - ``TrainSize``
     - ``50.0``
     - Percentage (0–100) of graphs used for training.  **Not** a 0–1
       fraction; ``80.0`` means 80 % training, 20 % validation/evaluation.
   * - ``BatchSize``
     - ``1``
     - Mini-batch size used during training and evaluation.
   * - ``Threads``
     - ``10``
     - Number of inter-operation threads for PyTorch.
   * - ``IntraThreads``
     - ``-1``
     - Number of intra-operation threads for PyTorch (``-1`` = PyTorch
       default).
   * - ``Training``
     - ``True``
     - Enable the training phase.
   * - ``Validation``
     - ``True``
     - Enable the validation phase.
   * - ``Evaluation``
     - ``True``
     - Enable the evaluation phase.
   * - ``ContinueTraining``
     - ``True``
     - Resume training from the last saved checkpoint.
   * - ``Targets``
     - ``[]``
     - List of target feature names used for GNN training.
   * - ``NumExamples``
     - ``3``
     - Number of graph examples logged per epoch.

Plot / Debug Settings
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Property
     - Default
     - Description
   * - ``nBins``
     - ``400``
     - Number of histogram bins for auto-generated kinematic plots.
   * - ``MaxRange``
     - ``400``
     - Upper range limit for auto-generated kinematic histograms.
   * - ``SetLogY``
     - ``False``
     - Use a logarithmic y-axis for auto-generated plots.
   * - ``DebugMode``
     - ``False``
     - Enable verbose debug output from the C++ engine.

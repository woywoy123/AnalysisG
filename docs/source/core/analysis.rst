Analysis
========

The Analysis class is the central orchestrator for running physics analyses in AnalysisG. It coordinates event processing, model training, metric evaluation, and result generation.

**File Location**: ``src/AnalysisG/core/analysis.pyx``

Overview
--------

The Analysis class manages the complete analysis workflow:

* Loading data from ROOT files
* Processing events through custom event classes
* Building graph representations
* Running selections and filters
* Training machine learning models
* Evaluating metrics
* Managing output and metadata
* Handling k-fold cross-validation

This class provides the main entry point for all AnalysisG analyses.

Basic Usage Pattern
-------------------

The typical workflow follows this pattern:

1. Create Analysis instance
2. Configure settings (output path, k-folds, etc.)
3. Add event type(s)
4. Add data samples
5. Optionally add graphs, selections, models, metrics
6. Run with ``Start()``

Quick Start Example
-------------------

.. code-block:: python

   from AnalysisG.core import Analysis, EventTemplate
   
   # Define your event class
   class MyEvent(EventTemplate):
       def __init__(self):
           super().__init__()
       
       def selection(self):
           leptons = [p for p in self.Particles if p.is_lepton()]
           return len(leptons) >= 2
   
   # Create and configure analysis
   analysis = Analysis()
   analysis.OutputPath = "./results"
   analysis.AddEvent(MyEvent(), "my_analysis")
   analysis.AddSamples("/path/to/data/*.root", "signal")
   
   # Run
   analysis.Start()

Core Methods
------------

AddEvent()
~~~~~~~~~~

.. py:method:: AddEvent(event, label)
   
   Register an event type for processing.
   
   :param event: EventTemplate instance defining event structure
   :type event: EventTemplate
   :param label: Unique identifier for this event type
   :type label: str
   
   **Purpose**: Tell Analysis how to process events from ROOT files.
   
   **Example**:
   
   .. code-block:: python
   
      from AnalysisG.events import BSM4Tops
      
      analysis.AddEvent(BSM4Tops(), "4top_events")

AddSamples()
~~~~~~~~~~~~

.. py:method:: AddSamples(path, label)
   
   Add ROOT files to process.
   
   :param path: Path to ROOT file(s), supports wildcards
   :type path: str
   :param label: Sample identifier (e.g., "signal", "background")
   :type label: str
   
   **Wildcard Support**: Use ``*`` for multiple files, ``?`` for single character.
   
   **Example**:
   
   .. code-block:: python
   
      # Single file
      analysis.AddSamples("/data/signal.root", "signal")
      
      # Multiple files with wildcard
      analysis.AddSamples("/data/ttbar_*.root", "ttbar")
      
      # All files in directory
      analysis.AddSamples("/data/backgrounds/*.root", "background")

AddGraph()
~~~~~~~~~~

.. py:method:: AddGraph(graph, label)
   
   Register a graph representation for events.
   
   :param graph: GraphTemplate instance
   :type graph: GraphTemplate
   :param label: Graph type identifier
   :type label: str
   
   **Purpose**: Enable graph-based analyses (e.g., for GNNs).
   
   **Example**:
   
   .. code-block:: python
   
      from AnalysisG.graphs import BSM4TopsGraph
      
      analysis.AddGraph(BSM4TopsGraph(), "event_graph")

AddSelection()
~~~~~~~~~~~~~~

.. py:method:: AddSelection(selection)
   
   Add a selection filter to the analysis.
   
   :param selection: SelectionTemplate instance
   :type selection: SelectionTemplate
   
   **Purpose**: Apply additional event-level or object-level selections.
   
   **Example**:
   
   .. code-block:: python
   
      from AnalysisG.selections.example import MET
      
      selection = MET()
      analysis.AddSelection(selection)

AddModel()
~~~~~~~~~~

.. py:method:: AddModel(model, optimizer, run_name)
   
   Add a machine learning model for training.
   
   :param model: ModelTemplate instance
   :type model: ModelTemplate
   :param optimizer: OptimizerConfig with training settings
   :type optimizer: OptimizerConfig
   :param run_name: Unique name for this training run
   :type run_name: str
   
   **Example**:
   
   .. code-block:: python
   
      from AnalysisG.models import GRIFT
      from AnalysisG.core import OptimizerConfig
      
      model = GRIFT()
      optimizer = OptimizerConfig()
      optimizer.learning_rate = 0.001
      optimizer.epochs = 100
      
      analysis.AddModel(model, optimizer, "grift_training")

AddModelInference()
~~~~~~~~~~~~~~~~~~~

.. py:method:: AddModelInference(model, run_name="run_name")
   
   Add a pre-trained model for inference only.
   
   :param model: Trained ModelTemplate instance
   :type model: ModelTemplate
   :param run_name: Run identifier
   :type run_name: str
   
   **Purpose**: Use trained model for predictions without training.

AddMetric()
~~~~~~~~~~~

.. py:method:: AddMetric(metric, model)
   
   Add an evaluation metric for model performance.
   
   :param metric: MetricTemplate instance
   :type metric: MetricTemplate
   :param model: Model to evaluate
   :type model: ModelTemplate
   
   **Example**:
   
   .. code-block:: python
   
      from AnalysisG.metrics import Accuracy
      
      metric = Accuracy()
      analysis.AddMetric(metric, model)

Start()
~~~~~~~

.. py:method:: Start()
   
   Execute the analysis workflow.
   
   **This method**:
   
   * Loads metadata (if FetchMeta enabled)
   * Processes all ROOT files
   * Applies event selection
   * Runs strategy methods
   * Trains models (if configured)
   * Evaluates metrics
   * Saves results to OutputPath
   * Shows progress bars for all operations
   
   **Example**:
   
   .. code-block:: python
   
      analysis.Start()  # Blocks until complete
      # Use Ctrl+C to interrupt if needed

Configuration Properties
------------------------

OutputPath
~~~~~~~~~~

.. py:attribute:: OutputPath
   
   Directory path for saving analysis results.
   
   :type: str
   :getter: Returns output path
   :setter: Sets output directory
   
   **Created Automatically**: Directory is created if it doesn't exist.
   
   **Contents**: Results, trained models, plots, metadata, etc.
   
   **Example**:
   
   .. code-block:: python
   
      analysis.OutputPath = "./output/4top_analysis"
      # Creates: output/4top_analysis/ with subdirectories

SumOfWeightsTreeName
~~~~~~~~~~~~~~~~~~~~

.. py:attribute:: SumOfWeightsTreeName
   
   Name of ROOT tree containing sum-of-weights information.
   
   :type: str
   :default: "sumWeights"
   :getter: Returns tree name
   :setter: Sets tree name
   
   **Purpose**: Used for normalization and cross-section calculations.
   
   **Example**:
   
   .. code-block:: python
   
      analysis.SumOfWeightsTreeName = "sumOfWeights"

kFolds
~~~~~~

.. py:attribute:: kFolds
   
   Number of folds for k-fold cross-validation.
   
   :type: int
   :default: 1 (no cross-validation)
   :getter: Returns number of folds
   :setter: Sets k for k-fold split
   
   **Purpose**: Enable cross-validation for model training.
   
   **Example**:
   
   .. code-block:: python
   
      # 5-fold cross-validation
      analysis.kFolds = 5

kFold
~~~~~

.. py:attribute:: kFold
   
   Which fold(s) to process.
   
   :type: int or list[int]
   :default: All folds
   :getter: Returns fold list
   :setter: Sets which folds to process
   
   **Example**:
   
   .. code-block:: python
   
      # Process only fold 0
      analysis.kFold = 0
      
      # Process folds 0 and 2
      analysis.kFold = [0, 2]

FetchMeta
~~~~~~~~~

.. py:attribute:: FetchMeta
   
   Whether to fetch and cache metadata from ROOT files.
   
   :type: bool
   :default: False
   :getter: Returns fetch status
   :setter: Enable/disable metadata caching
   
   **Purpose**: Cache expensive metadata operations across runs.
   
   **Example**:
   
   .. code-block:: python
   
      analysis.FetchMeta = True  # Enable for first run
      analysis.Start()
      # Subsequent runs reuse cached metadata

PreTagEvents
~~~~~~~~~~~~

.. py:attribute:: PreTagEvents
   
   Whether to pre-tag events before model training.
   
   :type: bool
   :default: False
   :setter: Enable pre-tagging
   
   **Purpose**: Prepare event labels before training.

GetMetaData
~~~~~~~~~~~

.. py:attribute:: GetMetaData
   
   Access cached metadata.
   
   :type: MetaLookup
   :getter: Returns metadata lookup object
   
   **Purpose**: Retrieve cross-sections, sum of weights, etc.

Complete Analysis Examples
---------------------------

Example 1: Simple Event Counting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import Analysis, EventTemplate
   
   class CountingEvent(EventTemplate):
       def __init__(self):
           super().__init__()
           self.n_jets = 0
           self.n_leptons = 0
       
       def selection(self):
           return True  # Accept all events
       
       def strategy(self):
           self.n_jets = sum(1 for p in self.Particles if p.is_jet())
           self.n_leptons = sum(1 for p in self.Particles if p.is_lepton())
   
   # Run
   analysis = Analysis()
   analysis.OutputPath = "./counting"
   analysis.AddEvent(CountingEvent(), "count")
   analysis.AddSamples("/data/*.root", "sample")
   analysis.Start()

Example 2: Multi-Sample Analysis with Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import Analysis, EventTemplate
   
   class TopPairEvent(EventTemplate):
       def __init__(self):
           super().__init__()
       
       def selection(self):
           # Dilepton selection
           electrons = [p for p in self.Particles 
                       if abs(p.pdgid) == 11 and p.pt > 25e3]
           muons = [p for p in self.Particles 
                   if abs(p.pdgid) == 13 and p.pt > 25e3]
           
           leptons = electrons + muons
           if len(leptons) < 2:
               return False
           
           # Jet requirement
           jets = [p for p in self.Particles 
                  if p.is_jet() and p.pt > 30e3]
           if len(jets) < 2:
               return False
           
           # B-jet requirement
           bjets = [j for j in jets if j.is_bjet()]
           if len(bjets) < 1:
               return False
           
           return True
   
   # Setup
   analysis = Analysis()
   analysis.OutputPath = "./ttbar_analysis"
   analysis.AddEvent(TopPairEvent(), "ttbar")
   
   # Add samples
   analysis.AddSamples("/data/ttbar/*.root", "signal")
   analysis.AddSamples("/data/zjets/*.root", "zjets_bkg")
   analysis.AddSamples("/data/wjets/*.root", "wjets_bkg")
   
   # Run
   analysis.Start()

Example 3: GNN Training with Cross-Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import Analysis, OptimizerConfig
   from AnalysisG.events import BSM4Tops
   from AnalysisG.graphs import BSM4TopsGraph
   from AnalysisG.models import GRIFT
   from AnalysisG.metrics import Accuracy
   
   # Create components
   event = BSM4Tops()
   graph = BSM4TopsGraph()
   model = GRIFT()
   metric = Accuracy()
   
   # Configure optimizer
   optimizer = OptimizerConfig()
   optimizer.learning_rate = 0.001
   optimizer.batch_size = 32
   optimizer.epochs = 100
   
   # Setup analysis
   analysis = Analysis()
   analysis.OutputPath = "./gnn_training"
   analysis.kFolds = 5  # 5-fold cross-validation
   
   # Add components
   analysis.AddEvent(event, "4top")
   analysis.AddGraph(graph, "event_graph")
   analysis.AddModel(model, optimizer, "grift_5fold")
   analysis.AddMetric(metric, model)
   
   # Add data
   analysis.AddSamples("/data/signal/*.root", "4top_signal")
   analysis.AddSamples("/data/background/*.root", "background")
   
   # Run training
   analysis.Start()

Example 4: Inference on New Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import Analysis
   from AnalysisG.events import BSM4Tops
   from AnalysisG.graphs import BSM4TopsGraph
   from AnalysisG.models import GRIFT
   
   # Load trained model
   model = GRIFT()
   model.load_state_dict("/path/to/trained_model.pt")
   
   # Setup analysis for inference
   analysis = Analysis()
   analysis.OutputPath = "./inference_results"
   analysis.AddEvent(BSM4Tops(), "4top")
   analysis.AddGraph(BSM4TopsGraph(), "graph")
   analysis.AddModelInference(model, "inference_run")
   
   # Add new data
   analysis.AddSamples("/data/new_sample/*.root", "new_data")
   
   # Run inference
   analysis.Start()

Best Practices
--------------

1. **Set OutputPath first**: Create output directory before adding components
2. **Add Event before Samples**: Event definition must exist before loading data
3. **Use meaningful labels**: Makes results easier to interpret
4. **Enable FetchMeta on first run**: Speeds up subsequent analyses
5. **Monitor progress bars**: They show detailed processing status
6. **Use Ctrl+C to stop**: Analysis handles interrupts gracefully
7. **Check output directory**: Results are automatically organized

Common Patterns
---------------

Multiple Event Types
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   analysis.AddEvent(SignalEvent(), "signal_def")
   analysis.AddEvent(BackgroundEvent(), "background_def")

Systematic Variations
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   event.Trees = ["nominal", "JET_JER_UP", "JET_JER_DOWN"]
   analysis.AddEvent(event, "with_systematics")

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

   for i, sample_path in enumerate(sample_paths):
       analysis.AddSamples(sample_path, f"sample_{i}")

Performance Tips
----------------

* **Use wildcards for multiple files**: More efficient than adding individually
* **Enable FetchMeta**: Cache metadata for faster subsequent runs
* **Process folds in parallel**: Run different k-folds on different machines
* **Optimize event selection**: Return False early in selection() for speed
* **Use strategy() wisely**: Only compute what you need

See Also
--------

* :doc:`event_template`: EventTemplate documentation
* :doc:`graph_template`: GraphTemplate documentation
* :doc:`model_template`: ModelTemplate documentation
* :doc:`../interfaces/overview`: Interface overview

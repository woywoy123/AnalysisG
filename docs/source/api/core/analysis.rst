Analysis Class
==============

.. currentmodule:: AnalysisG.core

The Analysis class is the central component of the AnalysisG framework, responsible for orchestrating 
the entire analysis workflow from sample loading to result generation.

Class Reference
---------------

.. class:: Analysis

   Main analysis orchestration class.

   The Analysis class manages the complete analysis pipeline including:
   
   - Sample loading and caching
   - Event template registration and processing
   - Graph generation for GNNs
   - Selection criteria application
   - Model training and inference
   - Metric calculation and reporting

   **Constructor**

   .. method:: __init__()

      Initialize a new Analysis instance.

      Creates the underlying C++ analysis object and initializes internal data structures.

   **Sample Management**

   .. method:: AddSamples(path: str, label: str)

      Add ROOT samples to the analysis.

      :param path: Path to ROOT file or directory containing ROOT files
      :param label: Label to identify this sample set (e.g., "signal", "background")
      :type path: str
      :type label: str

      Example:
      
      .. code-block:: python

         ana.AddSamples("/data/ttbar/*.root", "ttbar")
         ana.AddSamples("/data/zjets/*.root", "zjets")

   **Template Registration**

   .. method:: AddEvent(event_template: EventTemplate, label: str)

      Register an event template for processing.

      :param event_template: Event template instance defining event structure
      :param label: Unique label for this event template
      :type event_template: EventTemplate
      :type label: str

      The event template defines:
      
      - Which ROOT trees/branches to read
      - How to construct event objects
      - Event-level variables and calculations

      Example:
      
      .. code-block:: python

         class MyEvent(EventTemplate):
             def __init__(self):
                 super().__init__()
                 self.Tree = "nominal"
         
         ana.AddEvent(MyEvent(), "my_events")

   .. method:: AddGraph(graph_template: GraphTemplate, label: str)

      Register a graph template for GNN applications.

      :param graph_template: Graph template defining graph structure
      :param label: Unique label for this graph template
      :type graph_template: GraphTemplate
      :type label: str

      Graph templates define:
      
      - Node features and construction
      - Edge features and connectivity
      - Global graph features
      - Batching strategy

      Example:
      
      .. code-block:: python

         class MyGraph(GraphTemplate):
             def __init__(self):
                 super().__init__()
                 # Configure graph structure
         
         ana.AddGraph(MyGraph(), "particle_graph")

   .. method:: AddSelection(selection_template: SelectionTemplate)

      Register a selection template for event filtering.

      :param selection_template: Selection template defining selection criteria
      :type selection_template: SelectionTemplate

      Selection templates implement:
      
      - Event-level cuts
      - Particle-level requirements
      - Custom selection logic
      - Output variable calculations

      Example:
      
      .. code-block:: python

         class MySelection(SelectionTemplate):
             def Selection(self, event):
                 # Return True to keep event
                 return event.n_jets >= 4
         
         ana.AddSelection(MySelection())

   **Model Management**

   .. method:: AddModel(model_template: ModelTemplate, optimizer_config: OptimizerConfig, run_name: str)

      Add a model for training.

      :param model_template: Model template instance
      :param optimizer_config: Optimizer configuration (learning rate, etc.)
      :param run_name: Unique name for this training run
      :type model_template: ModelTemplate
      :type optimizer_config: OptimizerConfig
      :type run_name: str

      This method configures the model for training mode with:
      
      - Model architecture
      - Training hyperparameters
      - Optimization algorithm
      - Loss functions

      Example:
      
      .. code-block:: python

         from AnalysisG.core import OptimizerConfig
         
         model = MyModel()
         config = OptimizerConfig()
         config.learning_rate = 0.001
         
         ana.AddModel(model, config, "training_run_1")

   .. method:: AddModelInference(model_template: ModelTemplate, run_name: str = "run_name")

      Add a model for inference only (no training).

      :param model_template: Model template with loaded weights
      :param run_name: Name for this inference run
      :type model_template: ModelTemplate
      :type run_name: str

      Use this for:
      
      - Evaluating pre-trained models
      - Generating predictions
      - Performance testing

   .. method:: AddMetric(metric_template: MetricTemplate, model_template: ModelTemplate)

      Register evaluation metrics for a model.

      :param metric_template: Metric template defining metrics to compute
      :param model_template: Model to evaluate
      :type metric_template: MetricTemplate
      :type model_template: ModelTemplate

      Common metrics include:
      
      - Accuracy, precision, recall
      - ROC curves and AUC
      - Confusion matrices
      - Custom physics metrics

   **Execution**

   .. method:: Start()

      Execute the analysis.

      This method:
      
      1. Validates all registered components
      2. Loads and caches sample metadata
      3. Processes events through registered templates
      4. Generates graphs if configured
      5. Trains models or runs inference
      6. Calculates metrics
      7. Saves results

      The method blocks until analysis completes.

   **Properties**

   .. attribute:: OutputPath
      :type: str

      Directory path for output files.

      Set this before calling Start() to specify where results should be saved:
      
      .. code-block:: python

         ana.OutputPath = "/path/to/output"

   .. attribute:: FetchMeta
      :type: bool

      Whether to fetch metadata from PyAMI.

      When True, queries PyAMI for:
      
      - Cross-sections
      - Generator filters
      - Dataset information
      - Luminosity data

   .. attribute:: PreTagEvents
      :type: bool

      Whether to pre-tag events before processing.

      Pre-tagging can improve efficiency for:
      
      - Large datasets
      - Complex selections
      - Multiple iterations

Memory Management
-----------------

The Analysis class uses RAII principles through Cython:

- ``__cinit__``: Allocates C++ objects
- ``__dealloc__``: Cleans up C++ objects
- Automatic garbage collection of Python references

Thread Safety
-------------

The Analysis class is **not thread-safe**. Create separate instances for parallel analyses.

However, internal processing uses multi-threading for:

- Sample loading
- Event processing (when possible)
- Graph generation

See Also
--------

* :doc:`templates` - Template classes documentation
* :doc:`io` - ROOT I/O functionality  
* :doc:`../modules/optimizer` - Training optimization

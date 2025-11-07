Analysis Pipeline and Workflow
==============================

This document provides a comprehensive explanation of the AnalysisG analysis pipeline, including the complete workflow from data loading to training/inference, with detailed function call sequences and state management.

Overview
--------

The AnalysisG framework orchestrates a multi-stage pipeline for high-energy physics analysis and machine learning:

**Pipeline Stages**:

1. **Data Loading**: ROOT/HDF5 files → Event objects
2. **Event Selection**: Apply physics cuts and filters
3. **Graph Construction**: Events → Graph representations
4. **Training/Inference**: GNN models on graph datasets
5. **Evaluation**: Metrics computation and result storage

**Main Controller**: ``analysis`` class

Workflow Diagram
----------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                     User Configuration                       │
   │  • add_samples()                                            │
   │  • add_event_template()                                     │
   │  • add_selection_template()                                 │
   │  • add_graph_template()                                     │
   │  • add_model() / add_metric_template()                      │
   └──────────────────────┬──────────────────────────────────────┘
                          │
                          ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                      start() Method                          │
   │  Orchestrates entire pipeline                               │
   └──────────────────────┬──────────────────────────────────────┘
                          │
                          ▼
   ┌─────────────────────────────────────────────────────────────┐
   │                  1. Cache Validation                         │
   │  check_cache() - Check for pre-built data                   │
   └──────────────────────┬──────────────────────────────────────┘
                          │
                          ▼
   ┌─────────────────────────────────────────────────────────────┐
   │             2. Event Building (if needed)                    │
   │  build_events() - ROOT → Event objects                      │
   │   ├─ Parallel file reading                                  │
   │   ├─ Particle reconstruction                                │
   │   └─ Event serialization to HDF5                            │
   └──────────────────────┬──────────────────────────────────────┘
                          │
                          ▼
   ┌─────────────────────────────────────────────────────────────┐
   │             3. Selection Application                         │
   │  build_selections() - Apply physics cuts                    │
   │   ├─ Event filtering                                        │
   │   ├─ Custom selection logic                                 │
   │   └─ Cache filtered events                                  │
   └──────────────────────┬──────────────────────────────────────┘
                          │
                          ▼
   ┌─────────────────────────────────────────────────────────────┐
   │             4. Graph Construction                            │
   │  build_graphs() - Events → Graphs                           │
   │   ├─ Node/edge feature extraction                           │
   │   ├─ Edge connectivity computation                          │
   │   └─ Graph serialization                                    │
   └──────────────────────┬──────────────────────────────────────┘
                          │
                  ┌───────┴────────┐
                  ▼                ▼
   ┌──────────────────────┐  ┌──────────────────────┐
   │  5a. Training Mode   │  │ 5b. Inference Mode  │
   │  build_model_session()│  │ build_inference()   │
   │  ├─ K-fold CV        │  │ ├─ Load model       │
   │  ├─ Data batching    │  │ ├─ Run prediction   │
   │  ├─ Optimization     │  │ └─ Save outputs     │
   │  └─ Checkpointing    │  └─────────────────────┘
   └──────────┬───────────┘
              │
              ▼
   ┌─────────────────────────────────────────────────────────────┐
   │             6. Metrics Computation                           │
   │  build_metric() - Evaluate performance                      │
   │   ├─ Per-fold metrics                                       │
   │   ├─ Aggregated statistics                                  │
   │   └─ ROC curves, plots                                      │
   └─────────────────────────────────────────────────────────────┘

Analysis Class API
------------------

Public Methods
~~~~~~~~~~~~~~

add_samples()
^^^^^^^^^^^^^

**Signature**:

.. code-block:: cpp

   void add_samples(std::string path, std::string label)

**Purpose**: Register input ROOT files for processing

**Parameters**:

* ``path`` (string): File path or glob pattern (e.g., ``/data/*.root``)
* ``label`` (string): Sample identifier (e.g., ``"ttbar"``, ``"signal"``)

**Behavior**:

1. Resolves glob patterns to file list
2. Stores in ``file_labels`` map: ``label`` → ``path``
3. Files processed lazily on ``start()`` call

**Usage**:

.. code-block:: python

   from AnalysisG import Analysis
   
   ana = Analysis()
   ana.AddSamples("/data/ttbar/*.root", "ttbar")
   ana.AddSamples("/data/signal/*.root", "signal")

**Internal Storage**: ``std::map<std::string, std::string> file_labels``

add_event_template()
^^^^^^^^^^^^^^^^^^^^

**Signature**:

.. code-block:: cpp

   void add_event_template(event_template* ev, std::string label)

**Purpose**: Register custom event class for ROOT file parsing

**Parameters**:

* ``ev`` (event_template*): User-defined event class (inherits ``EventTemplate``)
* ``label`` (string): Event type identifier (must match sample label or be shared)

**Behavior**:

1. Clones event template: ``ev->clone()``
2. Stores in ``event_labels`` map: ``label`` → ``event_template*``
3. Template used to instantiate events during ``build_events()``

**Usage**:

.. code-block:: python

   from AnalysisG import EventTemplate
   
   class MyEvent(EventTemplate):
       def selection(self):
           return len([p for p in self.Particles if p.is_lep]) >= 2
   
   ana.AddEvent(MyEvent(), "ttbar")

**Internal Storage**: ``std::map<std::string, event_template*> event_labels``

**Key Point**: Event template defines:

* Particle registration (jets, leptons, etc.)
* ROOT branch mapping (via ``add_leaf()``)
* Selection logic (``selection()`` method)

add_selection_template()
^^^^^^^^^^^^^^^^^^^^^^^^^

**Signature**:

.. code-block:: cpp

   void add_selection_template(selection_template* sel)

**Purpose**: Register event filter for physics cuts

**Parameters**:

* ``sel`` (selection_template*): Selection criteria implementation

**Behavior**:

1. Clones selection: ``sel->clone()``
2. Stores in ``selection_names`` map: ``name`` → ``selection_template*``
3. Applied to all events during ``build_selections()``

**Usage**:

.. code-block:: python

   from AnalysisG import SelectionTemplate
   
   class FourTopSelection(SelectionTemplate):
       def selection(self, event):
           jets = [p for p in event.Particles if p.Type == "jet"]
           bjets = [p for p in jets if p.is_b]
           return len(jets) >= 4 and len(bjets) >= 2
   
   ana.AddSelection(FourTopSelection())

**Internal Storage**: ``std::map<std::string, selection_template*> selection_names``

add_graph_template()
^^^^^^^^^^^^^^^^^^^^

**Signature**:

.. code-block:: cpp

   void add_graph_template(graph_template* gr, std::string label)

**Purpose**: Register graph construction method for GNN input

**Parameters**:

* ``gr`` (graph_template*): Graph builder (inherits ``GraphTemplate``)
* ``label`` (string): Graph type identifier

**Behavior**:

1. Clones graph template: ``gr->clone()``
2. Stores in ``graph_labels`` map: ``(event_label, graph_name)`` → ``graph_template*``
3. Constructs graphs during ``build_graphs()``

**Usage**:

.. code-block:: python

   from AnalysisG import GraphTemplate
   
   class ParticleGraph(GraphTemplate):
       def build(self, event):
           # Build node features, edge connectivity
           pass
   
   ana.AddGraph(ParticleGraph(), "particle_graph")

**Internal Storage**: ``std::map<std::string, std::map<std::string, graph_template*>> graph_labels``

add_model()
^^^^^^^^^^^

**Signatures**:

.. code-block:: cpp

   void add_model(model_template* model, optimizer_params_t* op, std::string run_name)
   void add_model(model_template* model, std::string run_name)

**Purpose**: Register GNN model for training or inference

**Parameters**:

* ``model`` (model_template*): PyTorch model wrapper
* ``op`` (optimizer_params_t*): Training hyperparameters (learning rate, epochs, etc.)
* ``run_name`` (string): Training run identifier

**Behavior (with optimizer)**:

1. Clones model: ``model->clone()``
2. Stores in ``model_sessions``: ``(model*, optimizer_params_t*)``
3. Triggers training mode: ``build_model_session()``

**Behavior (without optimizer)**:

1. Stores in ``model_inference`` map
2. Triggers inference mode: ``build_inference()``

**Usage (Training)**:

.. code-block:: python

   from AnalysisG import ModelTemplate
   
   model = MyGNN()  # PyTorch model
   
   params = OptimizerParams()
   params.lr = 0.001
   params.epochs = 100
   params.optimizer = "Adam"
   
   ana.AddModel(model, params, "gnn_training")

**Usage (Inference)**:

.. code-block:: python

   model = MyGNN()
   model.Load("checkpoint.pt")  # Load pre-trained weights
   ana.AddModel(model, "inference_run")

**Internal Storage**: 

* Training: ``std::vector<std::tuple<model_template*, optimizer_params_t*>> model_sessions``
* Inference: ``std::map<std::string, model_template*> model_inference``

add_metric_template()
^^^^^^^^^^^^^^^^^^^^^

**Signature**:

.. code-block:: cpp

   void add_metric_template(metric_template* mx, model_template* mdl)

**Purpose**: Register evaluation metric for model performance

**Parameters**:

* ``mx`` (metric_template*): Metric implementation (accuracy, PageRank, etc.)
* ``mdl`` (model_template*): Model to evaluate

**Behavior**:

1. Associates metric with model
2. Stores in ``metric_names`` map
3. Computed during/after training via ``build_metric()``

**Usage**:

.. code-block:: python

   from AnalysisG.metrics import Accuracy
   
   metric = Accuracy()
   ana.AddMetric(metric, model)

**Internal Storage**: ``std::map<std::string, metric_template*> metric_names``

start()
^^^^^^^

**Signature**:

.. code-block:: cpp

   void start()

**Purpose**: Execute entire analysis pipeline

**Behavior**: Sequential execution of build stages:

1. ``check_cache()`` - Validate cached data
2. ``build_project()`` - Initialize output directories
3. ``build_events()`` - Process ROOT files
4. ``build_selections()`` - Apply event filters
5. ``build_graphs()`` - Construct graph representations
6. ``build_model_session()`` OR ``build_inference()`` - Train/infer
7. ``build_metric()`` - Compute evaluation metrics

**State Management**: Sets ``started = true`` to prevent re-execution

**Thread Safety**: Uses ``attach_threads()`` for parallel processing

**Usage**:

.. code-block:: python

   ana = Analysis()
   # ... configuration ...
   ana.Start()  # Blocks until complete

Private Methods (Internal Pipeline)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

check_cache()
^^^^^^^^^^^^^

**Purpose**: Validate existence of cached data to skip reprocessing

**Process**:

1. Check ``OutputPath`` for existing HDF5 files
2. Populate ``in_cache`` map: ``(sample, stage)`` → ``bool``
3. Set ``skip_event_build`` flags if events cached

**Cache Keys**:

* ``"events"`` - Reconstructed event objects
* ``"selections/{name}"`` - Filtered events
* ``"graphs/{graph_name}"`` - Constructed graphs

**Behavior**:

* If cached: Skip corresponding build stage
* If missing: Rebuild from source

build_project()
^^^^^^^^^^^^^^^

**Purpose**: Initialize output directory structure

**Creates**:

.. code-block:: text

   OutputPath/
   ├── events/
   │   └── {sample_label}/
   ├── selections/
   │   └── {selection_name}/{sample_label}/
   ├── graphs/
   │   └── {graph_name}/{sample_label}/
   ├── models/
   │   └── {run_name}/
   │       ├── checkpoints/
   │       └── logs/
   └── metrics/
       └── {metric_name}/

build_events()
^^^^^^^^^^^^^^

**Purpose**: Parse ROOT files into event objects

**Process**:

1. **Initialize I/O**: ``io* reader = new io()``
2. **For each sample**:
   
   a. Open ROOT file: ``reader->open_root(path)``
   b. Get TTree list: ``reader->list_root_trees()``
   c. **For each TTree**:
      
      * Get event template: ``event_labels[sample_label]``
      * **For each entry** in TTree:
        
        - Read branches: ``reader->read_root_branch()``
        - Build event: ``event->build(element_t*)``
        - Apply ``selection()``: Keep if ``true``
        - Serialize to HDF5: ``event->__reduce__()``
      
3. **Cache**: Write to ``OutputPath/events/{sample}.h5``

**Parallelization**: Multiple files processed concurrently

**Key Functions**:

* ``event_template::build(element_t*)`` - Construct event from ROOT data
* ``event_template::selection()`` - User-defined filter
* ``io::write_hdf5()`` - Serialize to disk

build_selections()
^^^^^^^^^^^^^^^^^^

**Purpose**: Apply additional event filters beyond event-level ``selection()``

**Process**:

1. **For each selection template**:
   
   a. Load cached events: ``io::read_hdf5()``
   b. **For each event**:
      
      * Apply ``selection_template::selection(event)``
      * Keep if returns ``true``
   
   c. Cache filtered events: ``OutputPath/selections/{name}/{sample}.h5``

**Use Case**: Multi-stage cuts (e.g., preselection → tight selection)

build_graphs()
^^^^^^^^^^^^^^

**Purpose**: Transform events into graph representations for GNNs

**Process**:

1. **For each graph template**:
   
   a. Load events (from selections or events)
   b. **For each event**:
      
      * Call ``graph_template::build(event)``
      * Extract node features (particle kinematics)
      * Compute edge connectivity (e.g., k-NN, fully connected)
      * Create edge features (DeltaR, invariant mass)
   
   c. Serialize graphs: ``graph_template::__reduce__()``
   d. Cache: ``OutputPath/graphs/{graph_name}/{sample}.h5``

**Parallelization**: Graphs built concurrently across samples

**Key Functions**:

* ``graph_template::build(event)`` - User-defined graph construction
* ``graph::edge_aggregation()`` - Compute edge features

build_model_session()
^^^^^^^^^^^^^^^^^^^^^

**Purpose**: Train GNN models with k-fold cross-validation

**Process**:

1. **Initialize DataLoader**:
   
   .. code-block:: cpp
   
      loader = new dataloader(settings)
      loader->add_graph_path(graph_paths)
      loader->set_kfolds(k)

2. **For each fold** (k-fold CV):
   
   a. **Initialize optimizer**: ``initialize_loop(optimizer*, k, model, config, report)``
      
      * Clone model for fold
      * Setup Adam/SGD optimizer
      * Reset learning rate scheduler
   
   b. **Training loop** (epochs):
      
      * **For each batch** in train set:
        
        - Load graphs: ``loader->next()``
        - Forward pass: ``model->forward(graphs)``
        - Compute loss: ``model->loss_fx(pred, truth)``
        - Backward: ``loss.backward()``
        - Update weights: ``optimizer->step()``
      
      * **Validation**:
        
        - Evaluate on validation set
        - Compute metrics
        - Save checkpoint if improved
   
   c. **Test evaluation**:
      
      * Load best checkpoint
      * Evaluate on test fold
      * Store results

3. **Aggregate results**: Average metrics across folds

**Parallelization**: Each fold trained in separate thread

**Static Method**: ``execution(model, settings, data, progress, output, content, msg)``

**Key Variables**:

* ``optimizer* trainer`` - Per-fold optimizer instances
* ``model_report* reports`` - Training metrics/logs
* ``std::vector<std::thread*> threads`` - Parallel fold training

build_inference()
^^^^^^^^^^^^^^^^^

**Purpose**: Run pre-trained model on new data

**Process**:

1. Load model weights: ``model->load(checkpoint_path)``
2. Set to evaluation mode: ``model->eval()``
3. **For each sample**:
   
   a. Load graphs: ``io::read_hdf5()``
   b. Batch graphs: ``dataloader::batch()``
   c. **For each batch**:
      
      * Forward pass: ``model->forward(graphs)``
      * Extract predictions
      * Store outputs: ``OutputPath/inference/{sample}.h5``

**No Gradient**: ``torch::no_grad()`` for efficiency

build_metric()
^^^^^^^^^^^^^^

**Purpose**: Compute evaluation metrics (accuracy, ROC, custom metrics)

**Process**:

1. **For each metric template**:
   
   a. Initialize: ``metric->initialize()``
   b. Load predictions + truth labels
   c. Compute metric: ``metric->compute(pred, truth)``
   d. Store results: ``OutputPath/metrics/{metric_name}/``

2. **Common metrics**:
   
   * Accuracy: ``(TP + TN) / (TP + TN + FP + FN)``
   * ROC curve: TPR vs FPR at varying thresholds
   * PageRank: Graph-based particle importance

**Static Method**: ``execution_metric(metric_t*, progress, msg)``

**Parallelization**: Metrics computed concurrently

Workflow Example (Complete)
----------------------------

Four-Top Analysis with GNN Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG import Analysis, EventTemplate, GraphTemplate, ModelTemplate
   from AnalysisG.metrics import Accuracy
   
   # 1. Define custom event class
   class FourTopEvent(EventTemplate):
       def selection(self):
           jets = [p for p in self.Particles if p.Type == "jet"]
           bjets = [p for p in jets if p.is_b]
           leptons = [p for p in self.Particles if p.is_lep]
           return len(jets) >= 4 and len(bjets) >= 2 and len(leptons) == 1
   
   # 2. Define graph construction
   class ParticleGraph(GraphTemplate):
       def build(self, event):
           # Nodes: all particles
           self.Nodes = event.Particles
           
           # Edges: k-NN connectivity (k=5)
           self.edge_index = self.knn(k=5)
   
   # 3. Initialize analysis
   ana = Analysis()
   ana.OutputPath = "./results"
   ana.kFolds = 5
   
   # 4. Add data
   ana.AddSamples("/data/ttbar/*.root", "ttbar")
   ana.AddSamples("/data/fourtop/*.root", "signal")
   
   # 5. Register templates
   ana.AddEvent(FourTopEvent(), "ttbar")
   ana.AddEvent(FourTopEvent(), "signal")
   ana.AddGraph(ParticleGraph(), "particle_graph")
   
   # 6. Setup model training
   from my_models import ParticleGNN
   model = ParticleGNN(input_dim=8, hidden_dim=64, output_dim=2)
   
   params = OptimizerParams()
   params.lr = 0.001
   params.epochs = 100
   params.optimizer = "Adam"
   params.weight_decay = 1e-4
   
   ana.AddModel(model, params, "fourtop_classifier")
   
   # 7. Add metrics
   ana.AddMetric(Accuracy(), model)
   
   # 8. Execute pipeline
   ana.Start()
   
   # Pipeline executes:
   # - check_cache(): No cache, rebuild all
   # - build_events(): Parse ROOT → 50,000 events
   # - build_graphs(): Events → graphs (5min)
   # - build_model_session(): 5-fold CV training (30min)
   #   * Fold 1: 80% train, 10% val, 10% test
   #   * ... (parallel execution)
   # - build_metric(): Compute accuracy per fold
   
   # Results saved to:
   # ./results/models/fourtop_classifier/
   #   ├── fold_0/checkpoint_best.pt
   #   ├── fold_1/checkpoint_best.pt
   #   └── ...
   # ./results/metrics/Accuracy/
   #   └── fold_results.json

**Output**:

.. code-block:: json

   {
     "accuracy": {
       "fold_0": 0.87,
       "fold_1": 0.89,
       "fold_2": 0.88,
       "fold_3": 0.86,
       "fold_4": 0.87,
       "mean": 0.874,
       "std": 0.011
     }
   }

Inference on New Data
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load pre-trained model
   model = ParticleGNN.Load("./results/models/fourtop_classifier/fold_0/checkpoint_best.pt")
   
   # Setup analysis for inference
   ana = Analysis()
   ana.OutputPath = "./inference_results"
   ana.AddSamples("/new_data/*.root", "unknown")
   ana.AddEvent(FourTopEvent(), "unknown")
   ana.AddGraph(ParticleGraph(), "particle_graph")
   ana.AddModel(model, "classify_unknown")  # No optimizer = inference mode
   ana.Start()
   
   # Pipeline executes:
   # - build_events(): Parse new ROOT files
   # - build_graphs(): Construct graphs
   # - build_inference(): Run model.forward(), save predictions
   
   # Results:
   # ./inference_results/inference/unknown.h5
   #   Columns: event_id, prediction, probability

Progress Monitoring
-------------------

Real-Time Progress APIs
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   
   # Start analysis in background thread
   ana.Start()  # Non-blocking if attach_threads() called
   
   while not ana.IsComplete()["all"]:
       progress = ana.Progress()
       print(f"Events: {progress['events'][0]:.1f}%")
       print(f"Graphs: {progress['graphs'][0]:.1f}%")
       print(f"Training: {progress['training'][0]:.1f}%")
       
       mode = ana.ProgressMode()
       print(f"Current: {mode['current']}")
       
       report = ana.ProgressReport()
       print(f"Status: {report['message']}")
       
       time.sleep(5)

**Methods**:

* ``progress()`` → ``std::map<std::string, std::vector<float>>``
  
  * Keys: ``"events"``, ``"graphs"``, ``"training"``, ``"metrics"``
  * Values: ``[current%, total_steps]``

* ``progress_mode()`` → ``std::map<std::string, std::string>``
  
  * ``"current"`` → Current pipeline stage name

* ``progress_report()`` → ``std::map<std::string, std::string>``
  
  * ``"message"`` → Detailed status message

* ``is_complete()`` → ``std::map<std::string, bool>``
  
  * Per-stage completion flags

Performance Considerations
--------------------------

Caching Strategy
~~~~~~~~~~~~~~~~

**First Run** (no cache):

* Total time: ~2 hours for 100k events
* Breakdown: Events (30min) + Graphs (60min) + Training (30min)

**Subsequent Runs** (with cache):

* Training only: ~30min
* Changes to model hyperparameters don't invalidate cache

**Cache Invalidation**:

* Changing event/selection templates → rebuild events
* Changing graph templates → rebuild graphs
* Model changes → no rebuild needed

Parallelization
~~~~~~~~~~~~~~~

**Multi-threading**:

* File reading: ``N_cores`` files processed simultaneously
* Graph construction: Parallel across samples
* K-fold training: Each fold in separate thread

**GPU Acceleration**:

* CUDA kernels for physics calculations (DeltaR, mass)
* PyTorch CUDA tensors for model forward/backward
* Batch size limited by GPU memory

**Bottlenecks**:

* Graph construction (CPU-bound)
* Disk I/O for large datasets
* GNN forward pass (GPU memory)

Memory Management
~~~~~~~~~~~~~~~~~

**Event Buffering**:

* Events loaded in chunks (default: 1000 events)
* HDF5 compression saves ~70% disk space

**Graph Caching**:

* Graphs cached to disk, not kept in memory
* DataLoader streams graphs in batches

**Model Training**:

* Gradient accumulation for large models
* Automatic mixed precision (AMP) support

See Also
--------

* :doc:`build_system` - CMake configuration and compilation
* :doc:`cpp_complete_reference` - C++ API reference
* :doc:`../core/analysis` - Analysis class documentation
* :doc:`../core/event_template` - Event template guide
* :doc:`../core/graph_template` - Graph construction

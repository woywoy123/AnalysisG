.. _quickstart:

====================
Quickstart
====================

This guide shows the basic steps to use AnalysisG.

Basic Workflow
--------------

The typical AnalysisG workflow involves:

1.  **Define Events**: Create event templates to structure High Energy Physics (HEP) data.
2.  **Define Graphs**: Convert events into graph structures.
3.  **Train GNN**: Train a Graph Neural Network (GNN) using the created graphs.
4.  **Analyze**: Apply the trained model or analysis selections to new data.


Example: Training a GNN Model
-----------------------------

Set up and run a GNN training process.

.. code-block:: python

     import AnalysisG
     from AnalysisG import Analysis
     from AnalysisG.core.lossfx import OptimizerConfig
     from AnalysisG.models import Grift # Example GNN model
     # Import your custom event and graph classes
     from .event import MyEvent
     from .graph import MyGraph

     # --- Configuration ---
     data_path = "./path/to/your/data/*.root"
     output_dir = "./MyAnalysisProject/"
     dataset_name = "my_dataset"

     event_def = MyEvent()
     graph_def = MyGraph()

     # Define the GNN model
     model = Grift()
     model.o_edge = {"edge_class": "CrossEntropyLoss"} # Output edge prediction
     model.o_graph = {"graph_class": "CrossEntropyLoss"} # Output graph prediction
     model.i_node = ["pt", "eta", "phi", "energy", "is_b"] # Input node features
     model.i_graph = ["met", "met_phi"] # Input graph features
     model.device = "cuda:0" # Use GPU 0 if available

     # Define the optimizer
     optimizer_cfg = OptimizerConfig()
     optimizer_cfg.Optimizer = "adam"
     optimizer_cfg.lr = 1e-4

     # --- Analysis Setup ---
     ana = Analysis()
     ana.ProjectName = "MyGNNTraining"
     ana.InputSample = dataset_name
     ana.AddSamples(data_path, dataset_name)
     ana.AddEvent(event_def, dataset_name)
     ana.AddGraph(graph_def, dataset_name)
     ana.AddModel(model, optimizer_cfg, "MyModel")

     # --- Training Settings ---
     ana.TrainingDataset = output_dir + "training_graphs.h5"
     ana.GraphCache = output_dir + "graph_cache/"
     ana.kFolds = 5 # Use 5-fold cross-validation
     ana.Epochs = 50
     ana.BatchSize = 32
     ana.TrainSize = 80 # Use 80% for training, 20% for validation
     ana.Targets = ["edge_class", "graph_class"] # Specify prediction targets
     ana.MaxRange = 1000 # Process first 1000 events for caching (optional)
     ana.Threads = 4 # Number of parallel threads

     # --- Run ---
     ana.Start()

Example: Applying a Trained Model (Inference)
---------------------------------------------

Load a trained model and apply it to new data.

.. code-block:: python

     import AnalysisG
     from AnalysisG import Analysis
     from AnalysisG.models import Grift # Use the same model structure
     # Import your custom event and graph classes
     from .event import MyEvent
     from .graph import MyGraph

     # --- Configuration ---
     data_path = "./path/to/new/data/*.root"
     output_dir = "./MyAnalysisProject/Inference/"
     dataset_name = "inference_data"
     model_checkpoint = "./MyAnalysisProject/MyGNNTraining/MyModel/state/epoch-50/kfold-1_model.pt" # Example path

     event_def = MyEvent()
     graph_def = MyGraph()

     # Define the GNN model structure (matching the trained one)
     model = Grift()
     model.o_edge = {"edge_class": "CrossEntropyLoss"}
     model.o_graph = {"graph_class": "CrossEntropyLoss"}
     model.i_node = ["pt", "eta", "phi", "energy", "is_b"]
     model.i_graph = ["met", "met_phi"]
     model.device = "cuda:0"
     model.checkpoint_path = model_checkpoint # Load trained weights

     # --- Analysis Setup ---
     ana = Analysis()
     ana.ProjectName = "MyGNNInference"
     ana.InputSample = dataset_name
     ana.AddSamples(data_path, dataset_name)
     ana.AddEvent(event_def, dataset_name)
     ana.AddGraph(graph_def, dataset_name)
     ana.AddModelInference(model, "MyInferenceRun") # Add model for inference

     # --- Inference Settings ---
     ana.GraphCache = output_dir + "graph_cache/" # Can reuse or create new cache
     ana.BatchSize = 64 # Can often use larger batch size for inference
     ana.Threads = 4

     # --- Run ---
     ana.Start()
     # Output ROOT files with predictions will be in ./MyAnalysisProject/Inference/MyInferenceRun/

Example: Running a Selection Analysis
-------------------------------------

Apply a predefined or custom selection to analyze events without GNNs.

.. code-block:: python

     import AnalysisG
     from AnalysisG import Analysis
     # Import a predefined event class or your custom one
     from AnalysisG.events.bsm_4tops import BSM4Tops
     # Import a predefined selection or your custom one
     from AnalysisG.selections.example.met import MET

     # --- Configuration ---
     data_path = "./path/to/your/data/*.root"
     output_dir = "./MySelectionAnalysis/"
     dataset_name = "selection_data"

     event_def = BSM4Tops() # Example: Use a predefined event structure
     selection_def = MET() # Example: Use a predefined MET selection

     # --- Analysis Setup ---
     ana = Analysis()
     ana.ProjectName = "METAnalysis"
     ana.InputSample = dataset_name
     ana.AddSamples(data_path, dataset_name)
     ana.AddEvent(event_def, dataset_name)
     ana.AddSelection(selection_def)

     # --- Selection Settings ---
     ana.Threads = 4
     ana.SaveSelectionToROOT = True # Save output histograms/trees to ROOT file
     ana.SelectionOutputDirectory = output_dir # Specify output location

     # --- Run ---
     ana.Start()

     # --- Access Results (Optional) ---
     # Results are saved to ROOT file, but can also be accessed programmatically:
     # print(f"Selection Passed Events: {selection_def.Passed}")
     # print(f"Selection Failed Events: {selection_def.Failed}")
     # Access histograms if defined in the selection, e.g.:
     # selection_def.hist_met.SaveFigure(output_dir + "met_histogram.png")


Next Steps
----------

For more details and advanced usage:

*   Read the :ref:`User Guide <user_guide/index>`
*   Explore the :ref:`API Reference <api_reference/index>`
*   Browse the :ref:`Tutorials <tutorials/index>`

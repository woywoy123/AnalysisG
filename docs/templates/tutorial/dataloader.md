.. _dataloader:

====================
Dataloader
====================

The `dataloader` is a central component in AnalysisG, responsible for managing and providing graph data during the training and evaluation of models.

Basic Concept
-----------

The Dataloader offers the following functionalities:

* **Efficient Data Batching**: Combining multiple graphs into batches
* **Data Caching**: Caching graph data for repeated use
* **K-Fold Cross-Validation**: Support for robust model validation
* **Filtering and Transforming**: Applying preprocessing to graphs


Example: Training a Model
-------------------------

Here's an example demonstrating how to set up and run a basic model training process using `AnalysisG`:

.. code-block:: python

    # --- Core Imports ---
    from AnalysisG.generators.analysis import Analysis
    from AnalysisG.graphs import bsm_4tops # Example graph implementation
    from AnalysisG.events.bsm_4tops import BSM4Tops # Example event implementation
    from AnalysisG.core.lossfx import OptimizerConfig
    from AnalysisG.models import RecursiveGraphNeuralNetwork # Example model
    from AnalysisG.core.io import IO

    # --- 1. Initialize Analysis ---
    ana = Analysis()
    ana.OutputPath = "./my_analysis_output/" # Define output directory
    ana.ProjectName = "BasicTraining"

    # --- 2. Define Data Samples ---
    # Replace with your actual sample paths and structure
    sample_path = "/path/to/your/samples/"
    sample_name = "ttbar_mc16"
    iox = IO()
    iox.Files = [sample_path + "file1.root", sample_path + "file2.root"] # Example files
    iox.Trees = ["nominal"] # Example tree name

    # --- 3. Configure Analysis Settings ---
    ana.AddSamples(iox.Files, sample_name)
    ana.AddEvent(BSM4Tops(), sample_name) # Use BSM4Tops event definition
    ana.AddGraph(bsm_4tops.GraphJets(), sample_name) # Use GraphJets graph definition

    ana.Threads = 4 # Number of parallel threads
    ana.BatchSize = 100 # Number of graphs per batch
    ana.TrainSize = 0.8 # Use 80% of data for training
    ana.Epochs = 10 # Number of training epochs
    ana.GraphCache = ana.OutputPath + "/GraphCache/" # Cache directory
    ana.Training = True # Enable training mode

    # --- 4. Configure Model ---
    model = RecursiveGraphNeuralNetwork()
    model.device = "cuda" # Use GPU if available, else "cpu"
    # Define input/output features based on your graph/event implementation
    model.i_node = ["eta", "phi", "pt", "energy"]
    model.o_node = ["node_prediction"] # Example output node feature

    # --- 5. Configure Optimizer ---
    optim = OptimizerConfig()
    optim.lr = 0.001 # Learning rate
    optim.weight_decay = 0.0001
    # Add other optimizer parameters as needed (e.g., optimizer type)

    # --- 6. Add Model to Analysis ---
    ana.AddModel(model, optim, "MyRGNModel") # Give your model a name

    # --- 7. Start Analysis (Training) ---
    print("Starting Analysis...")
    ana.Start()
    print("Analysis Complete.")


Further Resources
-----------------

* Full API documentation: :ref:`API-Dataloader <api_reference/dataloader>`
* Example code in the `/docs/examples` directory

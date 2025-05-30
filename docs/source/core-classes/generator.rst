.. _analysis-script:

Analysis Interface
------------------

The main interfacing class that automates and defines the workflow from MVA training, ROOT n-tuples production, GNN inference, sample generation and much more.

The C++ Interface
^^^^^^^^^^^^^^^^^

.. cpp:class:: analysis: public notification, public tools

   .. cpp:var:: settings_t m_settings

      A member struct varible used to control and specify runtime behaviour.

   .. cpp:function:: void add_samples(std::string path, std::string label)

      A function used to specify the directory of the ROOT samples used for the analysis.
      Accepted syntax for the path parameter is /path/<name>.root or /path/\*.root.
      The label parameter is useful when samples need to separated, but is optional.

   .. cpp:function::void add_selection_template(selection_template* sel)
    
      Specify a selection algorithm that does some further analysis on the ROOT samples.
      Can also be used to write separate ROOT files that are derived from the selection algorithm.

   .. cpp:function::void add_event_template(event_template* ev, std::string label)

      Specifies the event implementation to use for the framework.

   .. cpp:function::void add_graph_template(graph_template* gr, std::string label)

      Specifies the graph implementation for GNN related data structures. 
      These templates can be used for training and inference of data.

   .. cpp:function::void add_model(model_template* model, optimizer_params_t* op, std::string run_name)

      Add a GNN model to the collection and execute the training gien some run name and optimizer parameters.

   .. cpp:function::void add_model(model_template* model, std::string run_name)

      Simiar to the prior function, expect this interface is used to perform inference on data.
      This will produce ROOT based n-tuples, which contain output values specified by the model template.

   .. cpp:function::void start()

      Initialize the framework. 


.. py:class:: Analysis

   .. py:function:: AddSamples(str path, str label)

      A function used to assign a sample label (arbitrary name) to a particular dataset.

   .. py:function:: AddEvent(EventTemplate ev, str label)

      A function used to pass the event implementation to be used for subsequent compilations.

   .. py:function:: AddGraph(GraphTemplate ev, str label)

      A function used to tell the framework which graph implementation should be used.

   .. py:function:: AddSelection(SelectionTemplate selc)

      A function which adds any selection templates to the current analysis workflow.

   .. py:function:: AddModel(ModelTemplate model, OptimizerConfig op, str run_name)

      A function used to add a model to be trained, along with any optimizer hyperparameters that should be applied to the model.
      The additional `run_name` variable is used to create folders that contain the training output. 

   .. py:function:: AddModelInference(ModelTemplate model, str run_name = "run_name")

      A function used to add a trained model that should be used for inference studies.
      The `run_name` variable is used to generate folder structures for output ROOT files that hold model predictions.

   .. py:function:: Start()

   :ivar int BatchSize:
   
   :ivar bool FetchMeta:                      
   
   :ivar str BuildCache:                       
   
   :ivar bool PreTagEvents:
   
   :ivar bool SaveSelectionToROOT: 

   :ivar bool GetMetaData: Attempts to identify any meta-data associated with the input samples and queries PyAMI to match any results.

   :ivar list SumOfWeightsTreeName: Scans the ROOT file for possible sum of weights trees and histograms.

   :ivar str OutputPath: The output path of the results.

   :ivar int kFolds: Number of folds to train the model with.

   :ivar list kFold: A list of kfolds to train the model. Useful if not enough resources are available to do a full k-fold train at once.

   :ivar int Epochs: Number of epochs to train the model.

   :ivar int NumExamples: Number of test example to validate runtime of the model.

   :ivar str TrainingDataset: Path of the training set to use. If a value is given but no training set is available, the framework will dump a .h5 file.

   :ivar int TrainSize: Size of the training set in percentage.

   :ivar bool Training: Run the model over the training set.

   :ivar bool Validation: Run the model over the validation set in a k-fold training session.

   :ivar bool Evaluation: Run the model over the evaluation set.

   :ivar bool ContinueTraining: Continue training the model at the last known checkpoint.

   :ivar int nBins: Number of bins to plot the invariant mass metrics with.

   :ivar int Refresh: Progress bar refresh step.

   :ivar float MaxRange: Maximum range to plot the invariant mass metric plots.

   :ivar str VarPt: The transverse momentum variable string name to use for the invariant mass computation.

   :ivar str VarEta: The rapidity variable string name to use for the invariant mass computation.

   :ivar str VarPhi: The azimuthal angle variable string name to use for the invariant mass computation.

   :ivar str VarEnergy: The energy variable string name to use for the invariant mass computation.

   :ivar list Targets: The targets to plot (the output of the model) e.g. top_edge.

   :ivar bool DebugMode: Disables all threading.

   :ivar int Threads: Number of threads to run the framework over.

   :ivar str GraphCache: Specifies a directory in which graph_template outputs should be cached. This will generate .h5 files that can be reused.

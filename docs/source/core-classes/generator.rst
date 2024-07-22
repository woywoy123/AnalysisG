.. _analysis-script:

The Analysis Scripter
---------------------

This module is the main interface which combines all the `C++` and `cython` classes together.
The class is highly flexible and can be modified as needed by other frameworks. 

.. py:class:: Analysis

   .. py:function:: AddSamples(str path, str label)

   .. py:function:: AddEvent(EventTemplate ev, str label)

   .. py:function:: AddGraph(GraphTemplate ev, str label)

   .. py:function:: AddSelection(SelectionTemplate selc)

   .. py:function:: AddModel(ModelTemplate model, OptimizerConfig op, str run_name)

   .. py:function:: AddModelInference(ModelTemplate model, str run_name = "run_name")

   .. py:function:: Start()

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

Template Classes
================

AnalysisG uses a template-based design pattern where users extend base template classes 
to define their specific analysis behavior. This provides flexibility while maintaining 
a consistent interface.

Overview
--------

The framework provides several template classes:

- **EventTemplate**: Define event structure and behavior
- **ParticleTemplate**: Define particle properties and methods
- **GraphTemplate**: Define graph structures for GNNs
- **SelectionTemplate**: Define event selection criteria
- **MetricTemplate**: Define evaluation metrics
- **ModelTemplate**: Define machine learning models

All templates are implemented in Cython (``.pyx`` files) with corresponding header files (``.pxd``).

EventTemplate
-------------

.. currentmodule:: AnalysisG.core

.. class:: EventTemplate

   Base class for defining event structures.

   EventTemplate provides the interface between ROOT data and Python objects. Subclasses 
   define which ROOT trees, branches, and leaves to read, and how to construct event objects.

   **Constructor**

   .. method:: __init__()

      Initialize the event template.

      When subclassing, call super().__init__() first, then configure properties.

   **Properties**

   .. attribute:: Tree
      :type: str

      Name of the ROOT tree to read.

      Example:
      
      .. code-block:: python

         self.Tree = "nominal"

   .. attribute:: Trees
      :type: Union[str, list]

      Multiple tree names to read.

      Example:
      
      .. code-block:: python

         self.Trees = ["nominal", "systematic_up", "systematic_down"]

   .. attribute:: Branches
      :type: Union[str, list]

      Branch names to read from the tree.

      Example:
      
      .. code-block:: python

         self.Branches = ["jets_pt", "jets_eta", "jets_phi", "jets_E"]

   .. attribute:: index
      :type: Union[str, int]

      Event index or branch name containing indices.

   .. attribute:: weight
      :type: Union[str, float]

      Event weight or branch name containing weights.

   **Magic Methods**

   .. method:: __name__() -> str

      Return the name of the event template.

   .. method:: __hash__() -> int

      Return a hash of the event template configuration.

   .. method:: __eq__(other) -> bool

      Compare two event templates for equality.

   **Example Subclass**

   .. code-block:: python

      from AnalysisG.core import EventTemplate
      
      class MyEvent(EventTemplate):
          def __init__(self):
              super().__init__()
              
              # Configure tree reading
              self.Tree = "nominal"
              self.Branches = ["jets_pt", "jets_eta", "met_met"]
              
              # Define event-level variables
              self.weight = "event_weight"
              self.index = "event_number"
          
          def initialize(self):
              """Called after ROOT data is loaded."""
              # Perform event-level calculations
              self.ht = sum(self.jets_pt)
              self.n_jets = len(self.jets_pt)

ParticleTemplate
----------------

.. class:: ParticleTemplate

   Base class for defining particle structures.

   ParticleTemplate defines how individual particles are constructed from ROOT data 
   and what properties/methods they have.

   **Constructor**

   .. method:: __init__()

      Initialize the particle template.

   **Common Properties**

   Particle templates typically define:

   - Kinematic properties (pt, eta, phi, mass, energy)
   - Identification variables
   - Isolation variables
   - Truth matching information

   **Example Subclass**

   .. code-block:: python

      from AnalysisG.core import ParticleTemplate
      
      class Jet(ParticleTemplate):
          def __init__(self):
              super().__init__()
          
          @property
          def pt(self):
              return self._pt
          
          @property 
          def eta(self):
              return self._eta
          
          def is_btagged(self, wp='medium'):
              """Check if jet is b-tagged."""
              return self.btag_score > self.btag_wp[wp]

GraphTemplate
-------------

.. class:: GraphTemplate

   Base class for defining graph structures for GNNs.

   GraphTemplate is a base class that can be extended to define custom graph structures.
   The actual graph construction and feature extraction is handled by user implementations.

   **Constructor**

   .. method:: __init__()

      Initialize the graph template.

   **Properties**

   .. attribute:: index
      :type: int

      Graph index identifier (read-only).

   .. attribute:: Tree
      :type: str

      Name of the ROOT tree (read-only).

   .. attribute:: PreSelection
      :type: bool

      Whether pre-selection is enabled.

   **Example Usage**

   GraphTemplate is typically subclassed and used with the Analysis framework:

   .. code-block:: python

      from AnalysisG.core import GraphTemplate, Analysis
      
      class MyGraph(GraphTemplate):
          def __init__(self):
              super().__init__()
              self.PreSelection = True
      
      # Use with Analysis
      ana = Analysis()
      ana.AddGraph(MyGraph(), "my_graph")

SelectionTemplate
-----------------

.. class:: SelectionTemplate

   Base class for defining event selection criteria.

   SelectionTemplate is implemented in C++ with Cython bindings. Users typically extend  
   this class in C++ to implement custom selection logic, then access it from Python.

   **Constructor**

   .. method:: __init__(inpt=None)

      Initialize the selection template.

      :param inpt: Optional dictionary to restore state from pickle
      :type inpt: dict or None

   **Properties**

   .. attribute:: PassedWeights
      :type: dict

      Dictionary of passed event weights (read-only).

   **Methods**

   .. method:: dump(path="./pkl-data", name="")

      Save selection results to pickle file.

      :param path: Directory to save pickle file
      :param name: Name for the pickle file (defaults to selection name)
      :type path: str
      :type name: str

   .. method:: load(path="./pkl-data", name="")

      Load selection results from pickle file.

      :param path: Directory containing pickle file
      :param name: Name of the pickle file
      :type path: str
      :type name: str
      :return: Loaded SelectionTemplate or None if failed
      :rtype: SelectionTemplate or None

   .. method:: HashToWeightFile(hash_)

      Convert hash to weight file mapping.

      :param hash_: Hash or list/dict of hashes
      :return: List of (filename, weight) tuples
      :rtype: list

   .. method:: GetMetaData

      Get metadata associated with selection.

      :return: Dictionary mapping to Meta objects
      :rtype: dict

   **Example Usage**

   Selections are typically implemented in C++. For example, the MET selection:

   .. code-block:: python

      from AnalysisG.selections.example.met.met import MET
      from AnalysisG.core import Analysis
      
      # Use predefined selection
      selection = MET()
      
      # Add to analysis
      ana = Analysis()
      ana.AddSelection(selection)
      
      # After running, access results
      weights = selection.PassedWeights

MetricTemplate
--------------

.. class:: MetricTemplate

   Base class for defining evaluation metrics.

   MetricTemplate is used to evaluate model performance. It's typically extended
   in C++ with Python bindings.

   **Constructor**

   .. method:: __init__()

      Initialize the metric template.
      
      Note: Subclasses must set ``self.mtx`` to a C++ metric_template pointer.

   **Properties**

   .. attribute:: RunNames
      :type: dict

      Dictionary mapping run identifiers to names.

   .. attribute:: Variables
      :type: list

      List of variable names to track.

   **Methods**

   .. method:: Postprocessing()

      Optional post-processing hook called after metrics calculation.

   .. method:: InterpretROOT(path, epochs=[], kfolds=[])

      Interpret ROOT files containing metric data.

      :param path: Path to ROOT files (supports wildcards)
      :param epochs: List of epochs to process
      :param kfolds: List of k-fold indices to process
      :type path: str
      :type epochs: list
      :type kfolds: list

   **Example Usage**

   .. code-block:: python

      from AnalysisG.metrics import AccuracyMetric
      from AnalysisG.core import Analysis
      
      # Use predefined metric
      mx = AccuracyMetric()
      mx.RunNames = {"train": "Training", "valid": "Validation"}
      mx.Variables = ["accuracy", "loss"]
      
      # Add to analysis with model
      ana = Analysis()
      ana.AddMetric(mx, model)

ModelTemplate
-------------

.. class:: ModelTemplate

   Base class for defining machine learning models.

   ModelTemplate wraps models for use in the AnalysisG framework, typically PyTorch models.

   **Constructor**

   .. method:: __init__()

      Initialize the model template.

   **Properties**

   .. attribute:: o_graph
      :type: dict

      Output graph-level features mapping to loss functions.
      
      Example: ``{"signal": "CrossEntropyLoss"}``

   .. attribute:: o_node
      :type: dict

      Output node-level features mapping to loss functions.

   .. attribute:: o_edge
      :type: dict

      Output edge-level features mapping to loss functions.
      
      Example: ``{"top_edge": "CrossEntropyLoss"}``

   .. attribute:: i_graph
      :type: list

      Input graph-level feature names.
      
      Example: ``["met", "phi"]``

   .. attribute:: i_node
      :type: list

      Input node-level feature names.
      
      Example: ``["pt", "eta", "phi", "energy"]``

   .. attribute:: i_edge
      :type: list

      Input edge-level feature names.

   .. attribute:: device
      :type: str

      Device to run model on.
      
      Example: ``"cuda:0"`` or ``"cpu"``

   .. attribute:: checkpoint_path
      :type: str

      Path to model checkpoint file.

   .. attribute:: name
      :type: str

      Model name (defaults to class name).

   **Example Usage**

   .. code-block:: python

      from AnalysisG.models import Grift
      from AnalysisG.core import Analysis, OptimizerConfig
      
      # Create model instance
      model = Grift()
      model.name = "Grift-Model-v1"
      model.device = "cuda:0"
      
      # Configure outputs and loss functions
      model.o_edge = {"top_edge": "CrossEntropyLoss"}
      model.o_graph = {"signal": "CrossEntropyLoss"}
      
      # Configure inputs
      model.i_node = ["pt", "eta", "phi", "energy"]
      model.i_graph = ["met", "phi"]
      
      # Add to analysis for training
      config = OptimizerConfig()
      config.lr = 0.001
      
      ana = Analysis()
      ana.AddModel(model, config, "training_run")

Design Patterns
---------------

Inheritance Hierarchy
~~~~~~~~~~~~~~~~~~~~~

All templates follow this pattern:

1. Base template class (e.g., EventTemplate) defines interface
2. User creates subclass implementing specific behavior
3. Framework calls methods on subclass through base interface

This allows:

- Type safety through base classes
- Flexibility through inheritance
- Consistent API across different implementations

Template Methods
~~~~~~~~~~~~~~~~

Templates use the "template method" pattern:

- Base class defines algorithm skeleton
- Subclasses fill in specific steps
- Framework controls overall flow

Memory Management
~~~~~~~~~~~~~~~~~

Templates use Cython's memory management:

- Automatic reference counting
- C++ object lifecycle management
- Integration with Python garbage collection

See Also
--------

* :doc:`analysis` - Analysis class documentation
* :doc:`io` - ROOT I/O functionality
* :doc:`../modules/index` - Module documentation

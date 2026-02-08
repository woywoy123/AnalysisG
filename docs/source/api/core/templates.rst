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

   GraphTemplate specifies how to construct graphs from events and particles, 
   including node features, edge features, and connectivity.

   **Constructor**

   .. method:: __init__()

      Initialize the graph template.

   **Key Methods to Override**

   .. method:: Nodes(event) -> list

      Define which particles become nodes.

      :param event: Event object
      :return: List of particles to use as nodes
      :rtype: list

   .. method:: Edges(event) -> list

      Define edge connectivity.

      :param event: Event object
      :return: List of (source, target) tuples
      :rtype: list

   .. method:: NodeFeatures(particle) -> list

      Extract features from a particle/node.

      :param particle: Particle object
      :return: Feature vector
      :rtype: list

   .. method:: EdgeFeatures(source, target) -> list

      Calculate features for an edge.

      :param source: Source particle
      :param target: Target particle
      :return: Edge feature vector
      :rtype: list

   **Example Subclass**

   .. code-block:: python

      from AnalysisG.core import GraphTemplate
      
      class ParticleGraph(GraphTemplate):
          def __init__(self):
              super().__init__()
          
          def Nodes(self, event):
              # Use all jets as nodes
              return event.jets
          
          def Edges(self, event):
              # Connect all jets to all other jets
              edges = []
              for i, jet1 in enumerate(event.jets):
                  for j, jet2 in enumerate(event.jets):
                      if i != j:
                          edges.append((i, j))
              return edges
          
          def NodeFeatures(self, jet):
              # Return kinematic features
              return [jet.pt, jet.eta, jet.phi, jet.E]
          
          def EdgeFeatures(self, jet1, jet2):
              # Calculate delta R
              delta_r = ((jet1.eta - jet2.eta)**2 + 
                        (jet1.phi - jet2.phi)**2)**0.5
              return [delta_r]

SelectionTemplate
-----------------

.. class:: SelectionTemplate

   Base class for defining event selection criteria.

   SelectionTemplate implements the logic for deciding which events pass selection 
   and what variables to output.

   **Constructor**

   .. method:: __init__()

      Initialize the selection template.

   **Key Methods to Override**

   .. method:: Selection(event) -> bool

      Implement selection logic.

      :param event: Event object to evaluate
      :return: True if event passes selection
      :rtype: bool

   .. method:: Strategy(event)

      Perform calculations on selected events.

      :param event: Selected event object

      This method is called only for events passing Selection().

   **Example Subclass**

   .. code-block:: python

      from AnalysisG.core import SelectionTemplate
      
      class TTbarSelection(SelectionTemplate):
          def __init__(self):
              super().__init__()
          
          def Selection(self, event):
              # Require at least 4 jets
              if event.n_jets < 4:
                  return False
              
              # Require at least 1 lepton
              if event.n_leptons < 1:
                  return False
              
              # Require missing ET
              if event.met < 20:
                  return False
              
              return True
          
          def Strategy(self, event):
              # Calculate top mass candidates
              event.top_mass_candidates = []
              # ... reconstruction logic ...

MetricTemplate
--------------

.. class:: MetricTemplate

   Base class for defining evaluation metrics.

   MetricTemplate specifies how to evaluate model performance using various metrics.

   **Constructor**

   .. method:: __init__()

      Initialize the metric template.

   **Key Methods to Override**

   .. method:: Calculate(predictions, targets) -> dict

      Calculate metrics.

      :param predictions: Model predictions
      :param targets: Ground truth targets
      :return: Dictionary of metric name -> value
      :rtype: dict

   **Example Subclass**

   .. code-block:: python

      from AnalysisG.core import MetricTemplate
      import numpy as np
      
      class ClassificationMetrics(MetricTemplate):
          def __init__(self):
              super().__init__()
          
          def Calculate(self, predictions, targets):
              # Calculate accuracy
              correct = np.sum(predictions == targets)
              accuracy = correct / len(targets)
              
              # Calculate precision/recall
              # ... metric calculations ...
              
              return {
                  'accuracy': accuracy,
                  'precision': precision,
                  'recall': recall
              }

ModelTemplate
-------------

.. class:: ModelTemplate

   Base class for defining machine learning models.

   ModelTemplate wraps PyTorch models for use in the AnalysisG framework.

   **Constructor**

   .. method:: __init__()

      Initialize the model template.

   **Key Methods to Override**

   .. method:: forward(inputs) -> outputs

      Model forward pass.

      :param inputs: Input tensors
      :return: Output predictions

   **Example Subclass**

   .. code-block:: python

      from AnalysisG.core import ModelTemplate
      import torch.nn as nn
      
      class GNNClassifier(ModelTemplate):
          def __init__(self):
              super().__init__()
              
              # Define model architecture
              self.conv1 = GraphConv(4, 64)
              self.conv2 = GraphConv(64, 64)
              self.fc = nn.Linear(64, 2)
          
          def forward(self, data):
              x, edge_index = data.x, data.edge_index
              
              x = self.conv1(x, edge_index)
              x = torch.relu(x)
              x = self.conv2(x, edge_index)
              x = torch.relu(x)
              
              # Global pooling
              x = global_mean_pool(x, data.batch)
              
              # Classification
              x = self.fc(x)
              return x

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

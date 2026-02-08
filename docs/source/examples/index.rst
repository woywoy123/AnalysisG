Examples
========

This section provides practical examples of using AnalysisG.

Basic Examples
--------------

Reading ROOT Files
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import io
   
   # Open ROOT file
   file = io.open("data.root")
   tree = file.Get("nominal")
   
   # Read branches
   jets_pt = tree.Get("jets_pt")
   jets_eta = tree.Get("jets_eta")
   met = tree.Get("met_met")

Creating Event Template
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import EventTemplate, ParticleTemplate
   
   class Jet(ParticleTemplate):
       def __init__(self):
           super().__init__()
       
       @property
       def is_btagged(self):
           return self.btag_score > 0.8
   
   class MyEvent(EventTemplate):
       def __init__(self):
           super().__init__()
           self.Tree = "nominal"
           self.Branches = [
               "jets_pt", "jets_eta", "jets_phi", "jets_E",
               "met_met", "met_phi"
           ]
       
       def process(self):
           # Create jet objects
           self.jets = []
           for i in range(len(self.jets_pt)):
               jet = Jet()
               jet.pt = self.jets_pt[i]
               jet.eta = self.jets_eta[i]
               jet.phi = self.jets_phi[i]
               jet.E = self.jets_E[i]
               self.jets.append(jet)

Running Analysis
~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import Analysis
   
   # Create analysis
   ana = Analysis()
   ana.OutputPath = "output/"
   
   # Add samples
   ana.AddSamples("/data/ttbar/*.root", "ttbar")
   
   # Add event template
   ana.AddEvent(MyEvent(), "events")
   
   # Run
   ana.Start()

Advanced Examples
-----------------

Graph Neural Network
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import GraphTemplate
   import pyc.physics as phys
   
   class ParticleGraph(GraphTemplate):
       def __init__(self):
           super().__init__()
       
       def Nodes(self, event):
           return event.jets
       
       def Edges(self, event):
           edges = []
           for i in range(len(event.jets)):
               for j in range(i+1, len(event.jets)):
                   edges.append((i, j))
                   edges.append((j, i))
           return edges
       
       def NodeFeatures(self, jet):
           return [jet.pt, jet.eta, jet.phi, jet.E]
       
       def EdgeFeatures(self, jet1, jet2):
           dr = phys.DeltaR(jet1.eta, jet1.phi, jet2.eta, jet2.phi)
           return [dr]

Event Selection
~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import SelectionTemplate
   
   class TTbarSelection(SelectionTemplate):
       def __init__(self):
           super().__init__()
       
       def Selection(self, event):
           # At least 4 jets
           if len(event.jets) < 4:
               return False
           
           # At least 2 b-tagged jets
           n_btags = sum(1 for jet in event.jets if jet.is_btagged)
           if n_btags < 2:
               return False
           
           # Missing ET cut
           if event.met < 20:
               return False
           
           return True
       
       def Strategy(self, event):
           # Perform analysis on selected events
           event.ht = sum(jet.pt for jet in event.jets)

Neutrino Reconstruction
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.modules.nusol import NuSol
   
   solver = NuSol()
   
   for event in events:
       if len(event.leptons) != 1:
           continue
       
       lepton = event.leptons[0]
       
       # Solve for neutrino
       solutions = solver.Solve(
           lepton.pt, lepton.eta, lepton.phi, lepton.E,
           event.met, event.met_phi
       )
       
       # Select best solution
       if solutions:
           best = min(solutions, key=lambda s: s.chi2)
           event.neutrino = best

PyC High-Performance Computing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import pyc.physics as phys
   
   # Large batch of events
   n_events = 100000
   jet1_pt = torch.randn(n_events) * 100 + 150
   jet1_eta = torch.randn(n_events) * 2.5
   jet1_phi = torch.randn(n_events) * 3.14
   jet1_E = torch.randn(n_events) * 100 + 200
   
   jet2_pt = torch.randn(n_events) * 100 + 150
   jet2_eta = torch.randn(n_events) * 2.5
   jet2_phi = torch.randn(n_events) * 3.14
   jet2_E = torch.randn(n_events) * 100 + 200
   
   # Move to GPU
   device = torch.device('cuda')
   jet1_pt = jet1_pt.to(device)
   jet1_eta = jet1_eta.to(device)
   jet1_phi = jet1_phi.to(device)
   jet1_E = jet1_E.to(device)
   jet2_pt = jet2_pt.to(device)
   jet2_eta = jet2_eta.to(device)
   jet2_phi = jet2_phi.to(device)
   jet2_E = jet2_E.to(device)
   
   # Compute invariant mass for all events at once
   masses = phys.InvariantMass(
       jet1_pt, jet1_eta, jet1_phi, jet1_E,
       jet2_pt, jet2_eta, jet2_phi, jet2_E
   )
   
   # Results on GPU, extremely fast

Complete Analysis Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import (
       Analysis, EventTemplate, GraphTemplate, 
       SelectionTemplate, ModelTemplate
   )
   
   # 1. Define components
   class MyEvent(EventTemplate):
       # ... define event structure
       pass
   
   class MyGraph(GraphTemplate):
       # ... define graph structure
       pass
   
   class MySelection(SelectionTemplate):
       # ... define selection
       pass
   
   class MyModel(ModelTemplate):
       # ... define ML model
       pass
   
   # 2. Setup analysis
   ana = Analysis()
   ana.OutputPath = "results/"
   
   # 3. Add samples
   ana.AddSamples("/data/signal/*.root", "signal")
   ana.AddSamples("/data/background/*.root", "background")
   
   # 4. Configure pipeline
   ana.AddEvent(MyEvent(), "events")
   ana.AddSelection(MySelection())
   ana.AddGraph(MyGraph(), "graphs")
   ana.AddModel(MyModel(), optimizer_config, "training")
   
   # 5. Run
   ana.Start()

See Also
--------

* :doc:`../api/index` - Complete API reference
* :doc:`../introduction` - Framework introduction

Examples
========

This section provides practical examples of using AnalysisG based on actual usage patterns from the codebase.

Basic Examples
--------------

Reading ROOT Files with IO
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.io import IO
   
   # Create IO instance with file paths
   io = IO(["./samples/data.root"])
   
   # Specify trees and leaves to read
   io.Trees = ["nominal"]
   io.Leaves = ["jets_pt", "jets_eta", "met_met", "weight_mc"]
   
   # Iterate over events
   for event in io:
       print(event[b"nominal.jets_pt.jets_pt"])
       print(event[b"nominal.met_met.met_met"])

Reading Multiple ROOT Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.io import IO
   
   # Multiple files
   io = IO()
   io.Files = ["./samples/sample1.root", "./samples/sample2.root"]
   io.Trees = ["nominal", "truth"]
   io.Leaves = ["weight_pileup", "weight_mc", "met_phi", "children_pt"]
   
   # Scan available keys
   io.ScanKeys()
   
   for event in io:
       # Access data from different trees
       if b"truth.weight_mc.weight_mc" in event:
           weight = event[b"truth.weight_mc.weight_mc"]
       if b"nominal.children_pt.children_pt.children_pt" in event:
           pt = event[b"nominal.children_pt.children_pt.children_pt"]

Working with ParticleTemplate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.particle_template import ParticleTemplate
   import math
   
   class Particle(ParticleTemplate):
       def __init__(self):
           ParticleTemplate.__init__(self)
   
   # Create particle and set kinematics
   p = Particle()
   p.pt = 100.0
   p.eta = 0.5
   p.phi = 1.2
   p.e = 150.0
   
   # Access Cartesian coordinates (automatically calculated)
   print(p.px, p.py, p.pz)
   
   # Set PDG ID and charge
   p.pdgid = -11  # positron
   p.charge = 1
   
   # Check particle type
   print(p.symbol)  # "e"
   print(p.is_lep)  # True
   print(p.is_nu)   # False

Using EventTemplate
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.event_template import EventTemplate
   
   class MyEvent(EventTemplate):
       def __init__(self):
           super().__init__()
           # Specify which tree and branches to read
           self.Tree = "nominal"
           self.Branches = ["jets_pt", "jets_eta", "met_met"]
           # Map event properties to branch names
           self.weight = "event_weight"
           self.index = "event_number"

Running Analysis
~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG import Analysis
   from AnalysisG.selections.example.met.met import MET
   from AnalysisG.events.bsm_4tops.event_bsm_4tops import BSM4Tops
   
   # Create event and selection instances
   ev = BSM4Tops()
   sl = MET()
   
   # Setup analysis
   ana = Analysis()
   ana.AddSamples("./samples/dilepton/*", "tmp")
   ana.AddEvent(ev, "tmp")
   ana.AddSelection(sl)
   ana.DebugMode = False
   ana.SaveSelectionToROOT = True
   ana.Start()

Advanced Examples
-----------------

Using OptimizerConfig for Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG import Analysis
   from AnalysisG.core.lossfx import OptimizerConfig
   from AnalysisG.metrics import AccuracyMetric
   from AnalysisG.models import Grift
   
   # Create metric
   mx = AccuracyMetric()
   mx.RunNames = {"train": "Training", "valid": "Validation"}
   mx.Variables = ["accuracy", "loss"]
   
   # Create model
   gn = Grift()
   gn.name = "Grift-Model"
   gn.i_node  = ["pt", "eta", "phi", "energy"]
   gn.i_graph = ["met", "phi"]
   gn.device = "cuda:0"
   
   # Setup analysis for training
   ana = Analysis()
   ana.Threads = 4
   ana.BatchSize = 32
   ana.AddMetric(mx, gn)
   ana.GraphCache = "./cache/"
   ana.TrainingDataset = "./train.h5"
   ana.Validation = True
   ana.Training = True
   ana.Start()

OptimizerConfig Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.lossfx import OptimizerConfig
   
   config = OptimizerConfig()
   
   # Set optimizer type
   config.Optimizer = "Adam"  # or "SGD", "RMSprop", etc.
   config.Scheduler = "StepLR"
   
   # Learning rate settings
   config.lr = 0.001
   config.lr_decay = 0.95
   config.step_size = 10
   config.gamma = 0.1
   
   # Optimizer parameters
   config.weight_decay = 1e-5
   config.momentum = 0.9
   config.eps = 1e-8
   config.alpha = 0.99
   config.amsgrad = False

Particle Children and Parents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.particle_template import ParticleTemplate
   
   class Particle(ParticleTemplate):
       def __init__(self):
           ParticleTemplate.__init__(self)
   
   # Create parent and children particles
   parent = Particle()
   parent.pt = 200.0
   parent.eta = 0.0
   parent.phi = 0.0
   parent.e = 250.0
   
   child1 = Particle()
   child1.pt = 50.0
   child1.eta = 0.2
   child1.phi = 0.1
   child1.e = 60.0
   
   child2 = Particle()
   child2.pt = 80.0
   child2.eta = -0.3
   child2.phi = 0.2
   child2.e = 95.0
   
   # Build decay chain
   parent.Children += [child1]
   parent.Children.append(child2)
   child1.Parent.append(parent)
   child2.Parent.append(parent)
   
   # Access relationships
   print(len(parent.Children))  # 2
   print(child1 in parent.Children)  # True

Defining Variable Mappings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.particle_template import ParticleTemplate
   
   class Jet(ParticleTemplate):
       def __init__(self):
           ParticleTemplate.__init__(self)
           # Map object attributes to ROOT leaf names
           self.pt = "jets_pt"
           self.eta = "jets_eta"
           self.phi = "jets_phi"
           self.e = "jets_e"
   
   # Get the leaf mappings
   jet = Jet()
   leaves = jet.__getleaves__()
   print(leaves)  # {'pt': 'jets_pt', 'eta': 'jets_eta', ...}

Meta Data and PyAMI Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.io import IO
   
   io = IO("./sample.root")
   io.MetaCachePath = "./meta_cache"
   io.Trees = ["nominal"]
   io.Leaves = ["weight_mc"]
   io.EnablePyAMI = True
   
   # Get metadata
   meta_dict = io.MetaData()
   meta = list(meta_dict.values())[0]
   
   # Access metadata fields
   print(meta.dsid)
   print(meta.crossSection)
   print(meta.genFiltEff)
   print(meta.totalEvents)
   print(meta.generators)

See Also
--------

* :doc:`../api/index` - Complete API reference
* :doc:`../introduction` - Framework introduction
* :doc:`../installation` - Installation guide
   ana.Validation = True
   ana.Training = True
   ana.Start()

OptimizerConfig Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import OptimizerConfig
   
   config = OptimizerConfig()
   
   # Set optimizer type
   config.Optimizer = "Adam"  # or "SGD", "RMSprop", etc.
   config.Scheduler = "StepLR"
   
   # Learning rate settings
   config.lr = 0.001
   config.lr_decay = 0.95
   config.step_size = 10
   config.gamma = 0.1
   
   # Optimizer parameters
   config.weight_decay = 1e-5
   config.momentum = 0.9
   config.eps = 1e-8
   config.alpha = 0.99
   config.amsgrad = False

Particle Children and Parents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import ParticleTemplate
   
   class Particle(ParticleTemplate):
       def __init__(self):
           super().__init__()
   
   # Create parent and children particles
   parent = Particle()
   parent.pt = 200.0
   parent.eta = 0.0
   parent.phi = 0.0
   parent.e = 250.0
   
   child1 = Particle()
   child1.pt = 50.0
   child1.eta = 0.2
   child1.phi = 0.1
   child1.e = 60.0
   
   child2 = Particle()
   child2.pt = 80.0
   child2.eta = -0.3
   child2.phi = 0.2
   child2.e = 95.0
   
   # Build decay chain
   parent.Children += [child1]
   parent.Children.append(child2)
   child1.Parent.append(parent)
   child2.Parent.append(parent)
   
   # Access relationships
   print(len(parent.Children))  # 2
   print(child1 in parent.Children)  # True

Defining Variable Mappings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core import ParticleTemplate
   
   class Jet(ParticleTemplate):
       def __init__(self):
           super().__init__()
           # Map object attributes to ROOT leaf names
           self.pt = "jets_pt"
           self.eta = "jets_eta"
           self.phi = "jets_phi"
           self.e = "jets_e"
           self.btag = "jets_btag_score"
   
   # Get the leaf mappings
   jet = Jet()
   leaves = jet.__getleaves__()
   print(leaves)  # {'pt': 'jets_pt', 'eta': 'jets_eta', ...}

Meta Data and PyAMI Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from AnalysisG.core.io import IO
   
   io = IO("./sample.root")
   io.MetaCachePath = "./meta_cache"
   io.Trees = ["nominal"]
   io.Leaves = ["weight_mc"]
   io.EnablePyAMI = True
   
   # Get metadata
   meta_dict = io.MetaData()
   meta = list(meta_dict.values())[0]
   
   # Access metadata fields
   print(meta.dsid)
   print(meta.crossSection)
   print(meta.genFiltEff)
   print(meta.totalEvents)
   print(meta.generators)

See Also
--------

* :doc:`../api/index` - Complete API reference
* :doc:`../introduction` - Framework introduction
* :doc:`../installation` - Installation guide
               for j in range(i+1, len(event.jets)):
                   edges.append((i, j))

A Graph Neural Network Analysis Framework for High Energy Physics!
==================================================================

Abstract
--------
As the field of High Energy Particle Physics (HEPP) has begun exploring more exotic machine learning algorithms, such as Graph Neural Networks (GNNs).
Analyses commonly rely on pre-existing data science frameworks, including PyTorch, TensorFlow and Keras, to recast ROOT samples into the respective data structure.
This often results in tedious and computationally expensive co-routines to be written.
Community projects, such as UpROOT, Awkward, and Scikit-HEP are developing tools to address some of these problems.
For instance, in the context of Graph Neural Networks, converting non-monolithic data into graphs with edge, node and graph level feature becomes increasingly complex.

AnalysisG aims to address these remaining issues, by abstracting particles and events as Pythonic objects.
The framework initially populates particle and event objects, with user defined attributes that match strings contained in ROOT samples (trees/branches/leaves).
Particles living within the event can be retospectively matched with other particles to build complex decay chains, as is commonly done in truth matching.
Finalized event objects can be stored as HDF5 files, to prevent redundant reading and rebuilding of these objects.

To instantiate graphs, an additional template module interprets nominal Python functions as features derived from Particle and Event objects.
Implying that several graph types can be reinterpreted from the same Event/Particle object, with minimal code adjustments. 
Similar to the post event construction process, finalized graph types are stored as HDF5 files.

In the context of GNNs, the framework contains HEPP centric PyTorch based functions, written in C++ and native CUDA kernels.
Some of the functions included are; $\DeltaR$, Polar to Cartesian coordinate system transformations, Invariant Mass computations, edge/node single counting aggregation, analytical single/double neutrino reconstruction, and many more.


What is Analysis-G
------------------
Analysis-G is a Python package which aims to abstract and automate novel High Energy Particle Physics Analyses.
Since most of HEPP software relies on complicated ROOT files, the framework attempts to minimize the need for writing complicated and inefficient nested for loops to retrieve the content. 
This is achieved by defining particles and events as Python classes and assigning these attributes which allow the framework to infer and assign particles to events.
To bypass the speed limitations of Python, the underlying code has been written in C++ to assure scalability and minimize RAM usage when reading large number of ROOT files, without compromising the generality Python offers.

The framework places heavy emphasis on optimizing the interfacing between HEPP software and the machine learning community, which utilize vastly different data structure input. 
As such, once events are interpreted, they are stored in the HDF5 file format, thus removing the need for interacting with ROOT as early as possible.
A particular advantage of using the framework is made apparent in the context of constructing Graph like structures, which are required for training Graph Neural Networks. 
Since events and particles alike are abstracted into objects, their attributes can be utilized to populate Graphs with edge, node and global features in a fast and simplistic way, without resorting to repetitive code being written. 
This will be illustrated in dedicated tutorials of the docs.

Given the growing trend in machine learning, for instance Graph Neural Networks (GNNs), the framework aims to remain Analysis agnostic, such that several mutually exclusive ATLAS/Belle-II analyses can benefit from a single optimized framework. 

Core Packages in Analysis-G
---------------------------

- **EventTemplate:**
  A class which is to be inherited by custom event implementations. 
  This class interprets and retrieves data content found within ROOT files, and translates the ROOT structure in terms of particle content and per event attributes. 

- **ParticleTemplate:**
  A class used to define the most minimalistic representation of a particle expected to be found within the ROOT files. 
  Particles derived from this class will be generated on a per event basis such that the exact number of particles in each event will be made available.

- **GraphTemplate:**
  A template class which is used to specify which particles are to be used for Graph construction. 
  Graph features are assigned at a later stage of the Analysis pipeline, and will be discussed and illustrated in a dedicated tutorial. 

- **SelectionTemplate:**
  A class used to define a simplistic selection strategy, where particles and events (derived from **EventTemplates** and **ParticleTemplates**) will be made available for additional processing. 
  This class can be used to output ROOT n-tuples and then passed into some fitting tool, e.g. `TRexFitter` or `PyHF`

- **SampleTracer**
  An abstract module which can be utilized as a completely standalone package and can be integrated with your own framework. 
  This module aims to keep track of events and their original ROOT filename, and further permits fast event/graph retrieval. 
  The output of this module is a small HDF5 file, which only holds meta-data and mappings between ROOT files and their associated event indices. 

- **PyC (Python Custom)** 
  A completely detached package which implements high performance native C++ and CUDA code/kernels, which utilize the **PyTorch** API. 
  Several interfaces are implemented, namely switching from Cartesian to Polar coordinates, computing scalar invariant masses from particles, single/double neutrino reconstruction and many more. 


Getting Started with AnalysisG
------------------------------
.. toctree::
   :titlesonly:

   quick-start/installation.rst
   quick-start/getting-started.rst

Advanced Object Definitions
---------------------------
.. toctree::
   :titlesonly:

   quick-start/events.rst
   quick-start/particles.rst
   quick-start/selection.rst
   quick-start/graphs.rst

Analysis and Other Generators
-----------------------------
.. toctree::
   :titlesonly:

   generators/analysis.rst
   generators/sampletracer.rst
   generators/eventgenerator.rst
   generators/graphgenerator.rst
   generators/selectiongenerator.rst
   generators/samplegenerator.rst

Read and Writing (IO)
---------------------
.. toctree::
   :titlesonly:

   io/uproot.rst
   io/ntupler.rst

Machine Learning (Graph Neural Network)
---------------------------------------
.. toctree::
   :titlesonly:

   templates/features.rst
   machinelearning/optimizer.rst
   machinelearning/modelwrapper.rst
   gnn_training/schedule.rst

Condor and DAGMAN Submission Compilers
--------------------------------------
.. toctree::
   :titlesonly:
 
   submission/condor.rst

Plotting Functions
------------------
.. toctree::
   :titlesonly:
 
   plotting/plotting.rst

Tools Multi-Threading and Code Preservation
-------------------------------------------
.. toctree::
   :titlesonly:

   tools/tools.rst
   tools/multithreading.rst
   tools/code.rst

Data Types and Dictionary Mapping
---------------------------------
.. toctree::
   :titlesonly:

   cytypes/cytypes.rst

PyC, CUDA and C++ API via PyTorch
---------------------------------
.. toctree::
   :titlesonly:

   torch-extensions/main.rst
   torch-extensions/interface.rst

Analysis and Truth Studies Documentation
----------------------------------------
.. toctree::
   :titlesonly:
   :maxdepth: 0
 
   studies/truth-matching/main.rst 
   studies/strategies/main.rst
   studies/neutrino-studies/neutrino-reconstruction.rst

ROOT n-Tuple Event Implementations
----------------------------------
.. toctree::
   :maxdepth: 0
   :titlesonly:

   events/main.rst

Documentation and Codebase Status
---------------------------------

.. image:: https://readthedocs.org/projects/analysisg/badge/?version=latest
    :target: https://analysisg.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://github.com/woywoy123/AnalysisG/actions/workflows/AnalysisG_build.yml/badge.svg
   :target: https://github.com/woywoy123/AnalysisG/actions/workflows/AnalysisG_build.yml/badge.svg
   :alt: Build Status: AnalysisG

.. image:: https://github.com/woywoy123/AnalysisG/actions/workflows/pyc_build.yml/badge.svg
   :target: https://github.com/woywoy123/AnalysisG/actions/workflows/pyc_build.yml/badge.svg
   :alt: Build Status: Torch-Extension (pyc)

.. image:: https://github.com/woywoy123/AnalysisG/actions/workflows/release.yaml/badge.svg
   :target: https://github.com/woywoy123/AnalysisG/actions/workflows/release.yaml/badge.svg
   :alt: PyPI Release Build Status

.. image:: https://github.com/woywoy123/AnalysisG/actions/workflows/test.yml/badge.svg
   :target: https://github.com/woywoy123/AnalysisG/actions/workflows/test.yml/badge.svg
   :alt: Tox Testing Status

.. image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/woywoy123/6fee1eff8f987ac756a20133618659a1/raw/covbadge.json
   :target: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/woywoy123/6fee1eff8f987ac756a20133618659a1/raw/covbadge.json
   :alt: Coverage Report

.. image:: https://static.pepy.tech/badge/analysisg/month
   :target: https://pepy.tech/project/analysisg

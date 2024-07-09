A Graph Neural Network Analysis Framework for High Energy Physics!
==================================================================

Abstract
--------
As the field of High Energy Particle Physics (HEPP) has begun exploring more exotic machine learning algorithms, such as Graph Neural Networks (GNNs).
Analyses commonly rely on pre-existing data science frameworks, including PyTorch, TensorFlow and Keras, to recast ROOT samples into some respective data structure.
This often results in tedious and computationally expensive co-routines to be written.
Community projects, such as UpROOT, Awkward, and Scikit-HEP are developing tools to address some of these problems.
For instance, in the context of Graph Neural Networks, converting non-monolithic data into graphs with edge, node and graph level feature becomes increasingly complex.

AnalysisG aims to address these residual issues, by following a similar philosophy as *AnalysisTop*, whereby events and particles are treated as polymorphic objects.
The framework initially translate ROOT based n-tuples into user defined particle and event objects, where strings from ROOT samples (trees/branches/leaves) are mapped to their respective attributes.
Particles living within the event definition can be retrospectively matched with other particles to build complex decay chains, and subsequently used for truth matching studies and machine learning algorithms.

For Graph Neural Networks in particular, graph structures can be constructed via a template graph class, which will populate (graph, node and edge) feature tensors using the event and particle definitions using simple callable functions.
The resulting graphs can then be used for inference or supervised training sessions.

For preliminary cut based analyses, the framework offers selection templates, which take the prior event definitions as input and allows for detailed studies, which can then be exported into ROOT n-tuples.
Alternatively, the resulting selection templates can be assigned relevant attributes, which can be subsequently serialized and plotted as would be the case in dedicated truth studies.

To facilitate faster machine learning related training/inference in high energy particle physics, the framework exploits algorithms written entirely in C++ and native CUDA kernels.
These algorithms can be found within a self contained sub-package referred to as *pyc* and can be used outside of this framework. 
Some of the algorithms include; :math:`\Delta R`, Polar to Cartesian coordinate system transformations, Invariant Mass computations, edge/node single counting aggregation, analytical single/double neutrino reconstruction and many more.

Given the growing trend in machine learning, across multiple collaborations, the framework aims to remain Analysis agnostic, such that mutually exclusive ATLAS/Belle-II analyses can benefit from this framework. 

Core Modules and Languages used by AnalysisG
--------------------------------------------

To ensure optimal performance, the package uses C++ as the underlying language, but interfaces with Python using Cython.
Cython naturally interfaces with Python and provides minimal overhead in terms of multithreading limitations as would be the case of purely written Python code. 
Furthermore, Cython permits the sharing of C++ classes and pointers, meaning that memory is not unintentionally copied and introducing inefficiencies.

AnalysisG provides the following core modules that can be used in native C++, Cython and Python:

- **EventTemplate**: A template class used to specify to the framework which type of event and particle definitions to be used for the event.
- **ParticleTemplate**: A template class used in conjunction with **EventTemplate** to define the underlying particle type.
- **GraphTemplate**: A template class used to define the inclusive graph features, such as edge, node and global graph attributes. 
- **SelectionTemplate**: A template class for defining a customized event selection algorithm. 
- **Plotting**: A wrapper around **boost_histograms** and **mpl-hepp** that uses an object like syntax to define plotting routines.
- **io**: A Cython interface for the CERN ROOT C++ package, which centers around being simple to use and requiring as minimal syntax as possible to read ROOT n-tuples.
- **ModelTemplate**: A template class used to define machine learning algorithms.
- **Analysis**: The main analysis compiler used to define chains of actions to be performed given a user defined template class.
- **Tools**: A collection of tools that are used by the package.
- **pyc (Python CUDA)**: A completely detached package which implements high performance native C++ and CUDA code/kernels, utilizing the **PyTorch** API. 

DOCUMENTATION IS STILL UNDER CONSTRUCTION
-----------------------------------------

**The current documentation is being updated, since a lot of changes have been made from the prior version.**



Getting Started with AnalysisG
------------------------------
.. toctree::
   :titlesonly:

   getting-started/main.rst


Core Class Documentation
------------------------

.. toctree::
   :titlesonly:

   core-classes/main.rst


CUDA, C++ and pyc via LibTorch
------------------------------
.. toctree::
   :titlesonly:

   torch-extensions/main.rst
   torch-extensions/interface.rst

Event and Particle Implementations
----------------------------------

.. toctree::
   :titlesonly:

   mc16-events/main.rst



Analysis and Truth Studies Documentation
----------------------------------------
.. toctree::
   :titlesonly:
   :maxdepth: 1
 
   studies/main.rst 

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

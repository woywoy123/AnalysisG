.. AnalysisG documentation master file

AnalysisG Documentation
=======================

Welcome to the AnalysisG documentation! AnalysisG is a Graph Neural Network Analysis Framework for High Energy Physics.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   api/index
   examples/index
   modules/index

Introduction
============

As the field of High Energy Particle Physics (HEPP) has begun exploring more exotic machine learning algorithms, 
such as Graph Neural Networks (GNNs), analyses commonly rely on pre-existing data science frameworks, including 
PyTorch, TensorFlow and Keras, to recast ROOT samples into some respective data structure.

AnalysisG aims to address these issues by following a similar philosophy as *AnalysisTop*, whereby events and 
particles are treated as polymorphic objects. The framework initially translates ROOT based n-tuples into user 
defined particle and event objects, where strings from ROOT samples (trees/branches/leaves) are mapped to their 
respective attributes.

Core Features
-------------

* **EventTemplate**: A template class used to specify to the framework which type of event and particle definitions to be used for the event.
* **ParticleTemplate**: A template class used in conjunction with **EventTemplate** to define the underlying particle type.
* **GraphTemplate**: A template class used to define the inclusive graph features, such as edge, node and global graph attributes.
* **SelectionTemplate**: A template class for defining a customized event selection algorithm.
* **MetricTemplate**: A template class used to define evaluation metrics of some arbitrary machine learning algorithm.
* **Plotting**: A wrapper around **boost_histograms** and **mpl-hepp** that uses an object like syntax to define plotting routines.
* **io**: A Cython interface for the CERN ROOT C++ package.
* **MetaData**: This class exploits a modified version of PyAMI and performs additional DSID search and data-scraping.
* **ModelTemplate**: A template class used to define machine learning algorithms.
* **Analysis**: The main analysis compiler used to define chains of actions to be performed given a user defined template class.
* **pyc (Python CUDA)**: A completely detached package which implements high performance native C++ and CUDA code/kernels, utilizing the **LibTorch** API.

Quick Start
-----------

.. code-block:: python

   from AnalysisG.core import Analysis, EventTemplate, ParticleTemplate
   
   # Define your event and particle templates
   class MyEvent(EventTemplate):
       pass
   
   class MyParticle(ParticleTemplate):
       pass
   
   # Create and run analysis
   ana = Analysis()
   # ... configure your analysis

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

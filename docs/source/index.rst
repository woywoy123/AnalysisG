A Graph Neural Network Analysis Framework for High Energy Physics!
==================================================================

What is Analysis-G:
*******************
Analysis-G is a Python based package used to automate and stream-line the Particle Physics Analyses.
Specifically, it aims to reduce the entry barriers between the machine learning community and particle physics. 
The main motivation behind the package started with attempting to convert CERN-ROOT files into PyTorch Geometric Graph Data structures, which by no means was a trivial task to achieve. 
Most frameworks rely on nested for-loops and hard coded graph features that were specifically tailored towards their Analysis stream.

Given the growing trend in machine learning, for instance Graph Neural Networks (GNNs), the framework aims to remain Analysis agnostic, such that several mutually exclusive ATLAS/Belle-II analyses can benefit from a single optimized framework. 

Introduction
************
The aim of this package is to provide Particle Physicists with an intuitive interface to **Graph Neural Networks**, whilst remaining Analysis agnostic. 
Following a similar spirit to Analysis-Top, the physicist defines a custom event class and matches variables within ROOT files to objects that they are a representation of.
A simple example of this would be a particle, since these generally have some defining properties such as the four vector, mass, type, etc. 

From a technical point of view, the particle would be represented by some Python object, where attributes are matched to the ROOT leaf strings, such that framework can identify how variables are matched. 
A similar approach can be taken to construct event objects, where particle objects live within the event and are matched accordingly to any other particles e.g. particle truth matching. 
This hierarchical architecture allows for complex event definitions, first basic building blocks are defined and then matched according to some rule (see tutorial below).

To streamline the transition between ROOT and PyTorch Geometric (a Deep Learning framework for Graph Neural Networks), the framework utilizes event graph definitions.
These simply define which particles should be used to construct nodes on a PyTorch Geometric (PyG) Data object. Edge, Node and Graph features can be added separately as simple python functions (see tutorial below).
Post event graph construction, events are delegated to an optimization step, which trains a specified model with those graphs. 

To avoid having to deal with additional boiler plate book keeping code, the framework tracks the event to the originating ROOT file using a hashing algorithm. 
The hash is constructed by concatenating the directory, ROOT filename and event number into a single string and computing the associated hash. 
This ensures each event can be easily traced back to its original ROOT file. 

Index and Directories
*********************
.. toctree::
  quick-start/installation
  quick-start/getting-started.rst

Analysis and Truth Studies Documentation
****************************************
.. toctree::
  studies/analysis-strategies

Documentation and Codebase Status
*********************************
.. image:: https://readthedocs.org/projects/analysisg/badge/?version=latest
    :target: https://analysisg.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://github.com/woywoy123/AnalysisTopGNN/actions/workflows/test.yml/badge.svg?branch=master
   :target: https://github.com/woywoy123/AnalysisTopGNN/actions/workflows/test.yml/badge.svg?branch=master
   :alt: Build Status

.. image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/woywoy123/6fee1eff8f987ac756a20133618659a1/raw/covbadge.json
   :target: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/woywoy123/6fee1eff8f987ac756a20133618659a1/raw/covbadge.json
   :alt: Coverage Report

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
The framework initially translates ROOT based n-tuples into user defined particle and event objects, where strings from ROOT samples (trees/branches/leaves) are mapped to their respective attributes.
Particles living within the event definition can be retrospectively matched with other particles to build complex decay chains, and subsequently used for truth matching studies and machine learning algorithms.

For Graph Neural Networks in particular, graph structures can be constructed via template graphs, which will populate (graph, node and edge) feature tensors using the event and particle definitions using simple callable functions.
The resulting graphs can then be used for inference or supervised training sessions.

For preliminary cut based analyses, the framework offers selection templates, which take the prior event definitions as input and allows for detailed studies that can then be exported into ROOT n-tuples.
Alternatively, the resulting selection templates can be assigned relevant attributes, which can be subsequently serialized and plotted as would be the case in dedicated truth studies.

To facilitate faster machine learning related training/inference in high energy particle physics, the framework integrates algorithms written entirely in C++ and native CUDA kernels.
These algorithms can be found within a self contained sub-package referred to as *pyc* that can be used independently of the framework. 
These include; :math:`\Delta R`, Polar to Cartesian coordinate system transformations, Invariant Mass computations, edge/node single counting aggregation, analytical single/double neutrino reconstruction and many more.

Given the growing trend in machine learning, across multiple collaborations, the framework aims to remain Analysis agnostic, such that mutually exclusive ATLAS analyses can benefit from this framework. 

Core Modules and Languages used by AnalysisG
--------------------------------------------

To ensure optimal performance, the package uses C++ as the underlying language, but interfaces with Python using Cython.
Cython naturally interfaces with Python and provides minimal overhead in terms of multithreading limitations as would be the case of purely written Python code. 
Furthermore, Cython permits the sharing of C++ classes and pointers, meaning that memory is not unintentionally copied that introduce secondary inefficiencies.

Although the package is rather extensive, the following core modules provide a set of tools that might be helpful for other analyses:

- **EventTemplate**: A template class used to specify to the framework which type of event and particle definitions to be used for the event.
- **ParticleTemplate**: A template class used in conjunction with **EventTemplate** to define the underlying particle type.
- **GraphTemplate**: A template class used to define the inclusive graph features, such as edge, node and global graph attributes. 
- **SelectionTemplate**: A template class for defining a customized event selection algorithm.
- **MetricTemplate**: A template class used to define evaluation metrics of some arbitrary machine learning algorithm.
- **Plotting**: A wrapper around **boost_histograms** and **mpl-hepp** that uses an object like syntax to define plotting routines.
- **io**: A Cython interface for the CERN ROOT C++ package, which centers around being simple to use and requiring as minimal syntax as possible to read ROOT n-tuples (seriously, reading ROOT files can be done in 3 lines of code... check test/test_io.py).
- **MetaData**: This class exploits a modified version of PyAMI and performs additional DSID search and data-scraping.
- **ModelTemplate**: A template class used to define machine learning algorithms.
- **Analysis**: The main analysis compiler used to define chains of actions to be performed given a user defined template class.
- **Tools**: A collection of tools that are used by the package.
- **pyc (Python CUDA)**: A completely detached package which implements high performance native C++ and CUDA code/kernels, utilizing the **LibTorch** API. 

Prerequisites:
--------------
- **C++ Compiler:** A modern C++ compiler supporting C++20.
- **CMake:** Version 3.15 or higher.
- **Python:** Version 3.8 or higher.
- **Cython:** ```pip install cython```
- **HDF5:** (libhdf5)
- **ROOT:** A working installation of ROOT (https://root.cern/).
- **PyTorch:** (libtorch is automatically installed)
- **RapidJson:** (automatically installed)
  
Installation:
-------------
```bash
git clone https://github.com/username/AnalysisG.git
cd AnalysisG
mkdir build && cd build && cmake .. && make -jN && cmake ..
```

DOCUMENTATION IS STILL UNDER CONSTRUCTION
-----------------------------------------

**The current documentation is being updated, since a lot of changes have been made from the prior version.**
Although the documentation is rather incomplete, a lot of information can be found under the docs: https://analysisg.readthedocs.io/?badge=master

[![Documentation Status](https://readthedocs.org/projects/analysisg/badge/?version=master)](https://analysisg.readthedocs.io/?badge=master)

API Documentation
-----------------

In addition to the user documentation, comprehensive API documentation for all C++, CUDA, and Cython source files can be generated using Doxygen:

```bash
# Install doxygen (if not already installed)
sudo apt-get install doxygen graphviz

# Generate documentation
doxygen Doxyfile

# View documentation
firefox doxygen-docs/html/index.html
```

The generated documentation includes:
- Class hierarchies and relationships
- File and directory structure
- Source code browser with syntax highlighting
- Include dependency graphs
- All .h, .cxx, .cu, .cuh, .pyx, and .pxd files in src/AnalysisG
- Cython interface documentation showing Python-C++ bindings

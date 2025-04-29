Installation Guide
=================

This guide provides detailed instructions for installing the AnalysisG framework and its dependencies.

System Requirements
-----------------

* **Operating System**: Linux (Ubuntu 20.04+ or CentOS 7+ recommended)
* **RAM**: At least 8 GB (16+ GB recommended for large datasets)
* **Disk Space**: At least 10 GB for the framework and dependencies
* **GPU**: NVIDIA GPU with CUDA support recommended for training models

Dependencies
-----------

AnalysisG requires the following dependencies:

* C++ compiler with C++17 support (GCC 8+ or Clang 10+)
* CMake 3.12+
* ROOT 6.22+
* PyTorch 1.8.0+
* CUDA 10.2+ (for GPU support)
* Python 3.7+
* pybind11 2.6.0+
* Boost 1.71+
* HDF5 1.10+

Quick Installation
---------------

We provide a quick installation script for Ubuntu 20.04+:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/AnalysisG.git
   cd AnalysisG
   
   # Run the installation script
   ./scripts/install.sh

Step-by-Step Installation
----------------------

If the quick installation doesn't work for your system, follow these detailed steps:

1. **Install system dependencies**

   Ubuntu/Debian:
   
   .. code-block:: bash
   
      sudo apt update
      sudo apt install -y build-essential cmake git python3-dev python3-pip \
                         libboost-all-dev libhdf5-dev
   
   CentOS/RHEL:
   
   .. code-block:: bash
   
      sudo yum group install "Development Tools"
      sudo yum install -y cmake3 python3-devel boost-devel hdf5-devel

2. **Install ROOT**

   Follow the instructions at https://root.cern/install/ or use:
   
   .. code-block:: bash
   
      # Download ROOT
      wget https://root.cern/download/root_v6.24.06.Linux-ubuntu20-x86_64-gcc9.3.tar.gz
      tar -xzvf root_v6.24.06.Linux-ubuntu20-x86_64-gcc9.3.tar.gz
      
      # Set up ROOT environment
      source root/bin/thisroot.sh
      
      # Add to your .bashrc for future sessions
      echo 'source /path/to/root/bin/thisroot.sh' >> ~/.bashrc

3. **Install PyTorch**

   .. code-block:: bash
   
      # For CUDA support (adjust CUDA version as needed)
      pip3 install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
      
      # For CPU-only
      pip3 install torch==1.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

4. **Install pybind11**

   .. code-block:: bash
   
      pip3 install pybind11

5. **Build AnalysisG**

   .. code-block:: bash
   
      # Clone the repository if you haven't already
      git clone https://github.com/yourusername/AnalysisG.git
      cd AnalysisG
      
      # Create build directory
      mkdir build && cd build
      
      # Configure with CMake
      cmake ..
      
      # Build
      make -j4
      
      # Install
      make install

Environment Configuration
----------------------

After installation, you need to set up your environment:

.. code-block:: bash

   # Add to your .bashrc
   echo 'export ANALYSISG_PATH=/path/to/AnalysisG' >> ~/.bashrc
   echo 'export PYTHONPATH=$PYTHONPATH:$ANALYSISG_PATH/python' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ANALYSISG_PATH/lib' >> ~/.bashrc
   
   # Source your .bashrc
   source ~/.bashrc

Using Docker
----------

We also provide a Docker container with AnalysisG pre-installed:

.. code-block:: bash

   # Pull the Docker image
   docker pull analysisg/analysisg:latest
   
   # Run a container
   docker run -it --gpus all -v $(pwd):/workspace analysisg/analysisg:latest

Testing the Installation
---------------------

Verify your installation by running the tests:

.. code-block:: bash

   # Run C++ tests
   cd $ANALYSISG_PATH/build
   ctest
   
   # Run Python tests
   cd $ANALYSISG_PATH
   python3 -m pytest tests/python

Known Issues
----------

* **CUDA Compatibility**: Ensure your NVIDIA drivers are compatible with the CUDA version used by PyTorch
* **ROOT Version**: Some features may require specific ROOT versions
* **Memory Issues**: When processing large datasets, increase your swap space if needed

For detailed troubleshooting, please refer to the :doc:`../troubleshooting/index` section.

Additional Resources
-----------------

* :doc:`../tutorials/index`: Getting started tutorials
* :doc:`../examples/index`: Example projects using AnalysisG
* :doc:`../api/index`: API reference documentation


Introduction to AnalysisG
======================

AnalysisG is a physics analysis framework designed for high-energy particle physics, with a particular focus on optimizing workflows for analyzing data from the ATLAS experiment at CERN's Large Hadron Collider.

Framework Philosophy
-------------------

The framework is built on a few key principles:

1. **Graph-based approach**: Physics events are represented as graphs, where particles are nodes and their interactions are edges, enabling advanced machine learning techniques
2. **Modular design**: Components are loosely coupled, allowing you to use just what you need
3. **Performance-oriented**: C++ core with Python bindings for performance-critical operations

Key Features
-----------

* **Event handling**: Efficient reading and processing of event data from ROOT files
* **Physics object reconstruction**: Tools for particle identification and reconstruction
* **Graph-based analytics**: Specialized data structures for applying machine learning to physics data
* **Selection framework**: Define and apply physics region selections
* **Integrated machine learning**: Ready-to-use ML interfaces for physics analysis
* **Meta-data handling**: Tracking cross-sections, weights, and other experimental parameters

Framework Components
------------------

.. figure:: ../images/framework_structure.png
   :width: 600px
   :align: center
   :alt: AnalysisG Framework Components
   
   The key components and their relationships in the AnalysisG framework

The framework is organized into several key components:

* **Core**: The central analysis object and base functionality
* **Modules**: I/O, containers, meta-data handling, and other utilities
* **Events**: Event templates for different physics processes and experiments
* **Graphs**: Graph representations of physics events for machine learning
* **Selections**: Physics selection regions and criteria

Use Cases
--------

AnalysisG is particularly well-suited for:

* Top quark physics analyses
* BSM (Beyond Standard Model) searches
* Machine learning applications in particle physics
* Neutrino reconstruction
* Performance benchmarking of physics algorithms
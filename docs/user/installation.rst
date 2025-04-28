.. _installation:

Installation Guide
================

This guide covers the installation and setup process for AnalysisG.

Prerequisites
------------

AnalysisG requires the following dependencies:

* C++17 compatible compiler (GCC 9+, Clang 10+)
* CMake (version 3.14+)
* ROOT (version 6.20+)
* PyTorch (version 1.8+)
* Python 3.7+
* Boost (version 1.70+)

System Requirements
------------------

* **Memory**: 8GB minimum, 16GB+ recommended
* **Disk Space**: 5GB for a typical installation
* **OS**: Linux (Ubuntu 20.04+, CentOS 7+), macOS (10.15+)

Basic Installation
-----------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/woywoy123/AnalysisG.git
      cd AnalysisG

2. Create a build directory:

   .. code-block:: bash

      mkdir build && cd build

3. Configure with CMake:

   .. code-block:: bash

      cmake ..

4. Build the project:

   .. code-block:: bash

      make -j$(nproc)

5. Install (optional):

   .. code-block:: bash

      make install

Docker Installation
-----------------

For convenience, we provide Docker images with all dependencies pre-installed:

.. code-block:: bash

   docker pull woywoy123/analysisg:latest
   docker run -it --rm -v $PWD:/workspace woywoy123/analysisg:latest

Conda Installation
-----------------

We also provide a conda environment for easy setup:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate analysisg

Verifying Installation
--------------------

To verify the installation is working correctly:

.. code-block:: bash

   cd build
   ctest -V

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

1. **PyTorch not found during CMake configuration**

   Ensure that PyTorch is in your Python path. You can check with:

   .. code-block:: bash

      python -c "import torch; print(torch.__path__)"

2. **ROOT not found during compilation**

   Make sure ROOT is properly installed and sourced:

   .. code-block:: bash

      source /path/to/root/bin/thisroot.sh

3. **Compiler errors with C++17 features**

   Update your compiler to a newer version that fully supports C++17.

Getting Help
~~~~~~~~~~~

If you encounter issues not covered here:

1. Check the GitHub issues page for similar problems
2. Join our Slack channel for community support
3. Open a new issue with details about your environment and the exact error messages
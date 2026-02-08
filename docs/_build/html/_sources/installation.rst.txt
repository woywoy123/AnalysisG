Installation
============

Prerequisites
-------------

To ensure optimal performance, the package uses C++ as the underlying language, but interfaces with Python using Cython. 
Cython naturally interfaces with Python and provides minimal overhead in terms of multithreading limitations as would be 
the case of purely written Python code.

Required Dependencies
~~~~~~~~~~~~~~~~~~~~~

* Python >= 3.7
* CMake >= 3.23
* Cython
* scikit-build-core
* boost_histogram
* pyAMI-core
* mplhep
* pyyaml
* tqdm
* pwinput
* scipy
* h5py
* scikit-learn

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

* CUDA Toolkit (for GPU acceleration)
* Doxygen (for building documentation)
* Graphviz (for documentation diagrams)

Installation Methods
--------------------

From Source
~~~~~~~~~~~

To install AnalysisG from source:

.. code-block:: bash

   git clone https://github.com/woywoy123/AnalysisG.git
   cd AnalysisG
   pip install .

This will build the C++ extensions and install the Python package.

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development, install in editable mode:

.. code-block:: bash

   git clone https://github.com/woywoy123/AnalysisG.git
   cd AnalysisG
   pip install -e .

Verifying Installation
~~~~~~~~~~~~~~~~~~~~~~

After installation, verify that AnalysisG is properly installed:

.. code-block:: python

   import AnalysisG
   from AnalysisG.core import Analysis
   print("AnalysisG version:", AnalysisG.__version__ if hasattr(AnalysisG, '__version__') else "5.0")

Troubleshooting
---------------

Build Issues
~~~~~~~~~~~~

If you encounter build issues, ensure that:

1. CMake is properly installed and accessible
2. A compatible C++ compiler is available (GCC, Clang, or MSVC)
3. All dependencies are installed

CUDA Support
~~~~~~~~~~~~

For CUDA support, ensure that:

1. CUDA Toolkit is installed
2. The CUDA compiler (nvcc) is in your PATH
3. Your GPU drivers are up to date

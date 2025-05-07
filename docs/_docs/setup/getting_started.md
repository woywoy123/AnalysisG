Getting Started
===============

This guide helps you get AnalysisG installed and ready to use.

Prerequisites
-------------

*   **C++ Compiler:** A modern C++ compiler supporting C++17 (e.g., GCC >= 7, Clang >= 5).
*   **CMake:** Version 3.15 or higher.
*   **Python:** Version 3.8 or higher.
*   **ROOT:** A working installation of ROOT (https://root.cern/).
*   **PyTorch (Specific Version):** AnalysisG requires a specific ABI-compatible version of PyTorch. As mentioned in the main project documentation, if you need both PyTorch and AnalysisG in the same Python environment, install the correct version:

    .. code-block:: bash

       pip install "torch==2.4.0+cpu.cxx11.abi" -f https://download.pytorch.org/whl/torch_stable.html
       # Or for CUDA 12.1 support:
       # pip install "torch==2.4.0+cu121.cxx11.abi" -f https://download.pytorch.org/whl/torch_stable.html

    *Note: The exact required version might change. Refer to the project's main README or installation guide for the latest compatibility information.*

Installation
------------

(Instructions on how to build/install AnalysisG would go here. This typically involves cloning the repository and using CMake.)

.. code-block:: bash

   git clone https://github.com/your-repo/AnalysisG.git
   cd AnalysisG
   mkdir build && cd build
   cmake ..
   make install
   # Additional steps for setting up the Python environment might be needed.


Basic Workflow Overview
-----------------------

1.  **Define Event Structure:** Use `EventTemplate` and `ParticleTemplate` to describe the physics event and its constituent particles based on your input data format (e.g., ROOT n-tuples).
2.  **Define Graph Construction:** Use `GraphTemplate` to specify how to build graphs from the event data, including node features, edge features, graph features, and truth information.
3.  **Configure Analysis:** Set up an `analysis` object, specifying input files, event/graph definitions, and desired output.
4.  **Run Analysis:** Execute the analysis to generate graphs or perform other tasks like selection.
5.  **(Optional) Train GNN:** Use the generated graphs with PyTorch to train a Graph Neural Network. AnalysisG provides tools (`dataloader`, `optimizer`, `model_template`) to facilitate this.

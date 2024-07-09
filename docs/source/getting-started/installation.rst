Installation of AnalysisG
^^^^^^^^^^^^^^^^^^^^^^^^^

The project has a few dependency, these are listed below:

.. code-block:: console

   ROOT CERN (root-framework in snap store)
   HDF5 (libhdf5-dev)
   RapidJSON
   LibTorch (PyTorch C++ API)

For the **libtorch** and **rapidjson** packages, the provided CMake will automatically pull the relevant packages using **FetchContent**.

To install the package:

.. code-block:: console

   # clone the repository
   git clone https://github.com/woywoy123/AnalysisG.git
   cd AnalysisG
   pip install . 

   # Alternatively using only cmake 
   mkdir build && cd build && cmake ..

   # the last cmake call updates the package 
   make -j12 && cmake .. 

The installation process might be computationally heavy since it also compiles the **pyc** package and especially for CUDA kernels, requires a lot more compilation steps.

Installation of AnalysisG
^^^^^^^^^^^^^^^^^^^^^^^^^

The project has a few dependency, these are listed below:

.. code-block:: console

   ROOT CERN (root-framework in snap store)
   HDF5 (libhdf5-dev)
   RapidJSON - (installed using cmake automatically)
   LibTorch (PyTorch C++ API) - (installed using cmake automatically)

For the **libtorch** and **rapidjson** packages, the provided CMake will automatically pull the relevant packages using **FetchContent**.

To install the package:

.. code-block:: console

   # clone the repository
   git clone https://github.com/woywoy123/AnalysisG.git
   cd AnalysisG
   mkdir build && cd build 

   # the last cmake call scans for the site-package directory of pip and simply copies the build directory to the site-package path
   cmake .. && make -j12 && cmake .. 

The installation process might be computationally heavy since it also compiles the **pyc** package and especially for CUDA kernels, requires a lot more compilation steps.

**WARNING**
If running a Python script requires both torch and the analysis framework, please uninstall torch completely using pip and install the ABI compatible version.
The framework is currently compiled with torch 2.4.0-cu121, so to make torch compatible with the framework install the following torch version:

.. code-block:: console

   pip install pip install "torch==2.4.0+cpu.cxx11.abi" -i https://download.pytorch.org/whl/

Note the `cxx11.abi` extension.
This is important since most of the wheels provided by torch are all compiled using pre-ABI and cause missing symbols. 
See the Github issue; https://github.com/pytorch/pytorch/issues/51039



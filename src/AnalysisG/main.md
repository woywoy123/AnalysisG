# A Graph Neural Network Analysis Framework for High Energy Physics
## Abstract
As the field of High Energy Particle Physics (HEPP) has begun exploring more exotic machine learning algorithms, such as Graph Neural Networks (GNNs), analyses commonly rely on pre-existing data science frameworks to recast ROOT samples into appropriate data structures. This often results in tedious and computationally expensive co-routines to be written. Community projects like UpROOT, Awkward, and Scikit-HEP are developing tools to address some of these challenges.
For instance, converting non-monolithic data into graphs with edge, node, and graph-level features becomes increasingly complex when using Graph Neural Networks (GNNs). 
**AnalysisG** aims to address these residual issues by following a similar philosophy as *AnalysisTop*, treating events and particles as polymorphic objects. The framework translates ROOT-based n-tuples into user-defined particle and event objects, mapping strings from ROOT samples (trees/branches/leaves) to their respective attributes.
Particles within the event definition can be retrospectively matched with other particles to build complex decay chains, which are then used for truth matching studies and machine learning algorithms. For GNNs specifically, graph structures can be constructed using a template graph class that populates (graph, node, and edge) feature tensors based on the event and particle definitions.
The resulting graphs can be utilized for inference or supervised training sessions. For preliminary cut-based analyses, the framework offers selection templates, which take prior event definitions as input to allow detailed studies, which can then be exported into ROOT n-tuples. Alternatively, these selection templates can be assigned relevant attributes, which can subsequently be serialized and plotted as in dedicated truth studies.
To accelerate machine learning-related training/inference in high energy particle physics, the framework leverages algorithms written entirely in C++ and native CUDA kernels. These algorithms are contained within a self-contained sub-package referred to as *pyc*. Some of these algorithms include ΔR calculations, polar-to-Cartesian coordinate system transformations, invariant mass computations, edge/node single counting aggregation, analytical single/double neutrino reconstruction, and more.
Given the growing trend in machine learning across multiple collaborations, **AnalysisG** aims to remain analysis-agnostic, enabling mutually exclusive ATLAS/Belle-II analyses to benefit from this framework.

## Installation Prerequisites
The project has several dependencies:
1. **ROOT CERN** (available in the Snap Store)
2. **HDF5** (libhdf5-dev)
3. **RapidJSON**
4. **LibTorch** (PyTorch C++ API)
For **libtorch** and **rapidjson**, the provided CMake will automatically fetch the relevant packages using **FetchContent**.
To install the package:
```bash
# Clone the repository
git clone https://github.com/woywoy123/AnalysisG.git
cd AnalysisG
mkdir build && cd build
# The last cmake call scans for the site-package directory of pip and simply copies the build directory to the site-package path
cmake .. && make -j12 && cmake ..
```

## Warning
If running a Python script requires both torch and the analysis framework, please uninstall torch completely using pip and install the ABI-compatible version. The framework is currently compiled with torch 2.4.0-cu121, so to make torch compatible with the framework, install the following torch version:
```bash
pip install pip install "torch==2.4.0+cpu.cxx11.abi" -i https://download.pytorch.org/whl/
```
Note the **cxx11.abi** extension is important since most of the wheels on PyPI are built without it, which can lead to compatibility issues.

## Documentation
The documentation section will be updated as work progresses.


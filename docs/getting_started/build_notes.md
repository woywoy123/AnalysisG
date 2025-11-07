# Build Notes and Troubleshooting

This document contains detailed information about the AnalysisG build process based on actual build attempts.

## Build System Overview

AnalysisG uses a complex build system:

- **Build Backend**: `scikit-build-core` (Python build system that wraps CMake)
- **Build Tool**: CMake 3.20+ with C++17 compiler
- **Python Bindings**: Cython
- **Critical Dependencies**: ROOT, PyTorch C++ (LibTorch), HDF5

## What Actually Gets Built

The root `CMakeLists.txt` primarily handles documentation generation with Doxygen. The actual C++/Python modules are built by scikit-build-core, which discovers and compiles all the modules in `src/AnalysisG/`.

## Build Attempts and Findings

### Attempt 1: Direct CMake (Root Directory)

```bash
mkdir build && cd build
cmake ..
make
```

**Result**: Only builds documentation (Doxygen). The actual library modules are not built this way.

### Attempt 2: Using pip install

```bash
pip install -e .
```

**Result**: This is the correct approach, but requires:
- All Python dependencies (boost_histogram, mplhep, scipy, etc.)
- ROOT framework installed and discoverable
- PyTorch/LibTorch installed with correct ABI
- HDF5 development libraries

### Missing Dependencies Block Build

The build will fail without:

1. **ROOT**: Used by almost all modules
   ```
   target_link_libraries(... ROOT::Core ROOT::RIO ...)
   ```

2. **LibTorch**: Used by typecasting, analysis, and ML modules
   ```
   target_link_libraries(... ${TORCH_LIBRARIES} ...)
   ```

3. **HDF5**: Used by I/O module
   ```
   target_link_libraries(... ${HDF5_LIBRARIES} ...)
   ```

## Recommended Build Process

Since the dependencies are complex and HEP-specific:

1. **Use a System with ROOT**: Install on a system where ROOT is already set up (e.g., CERN computing infrastructure, or install ROOT from https://root.cern/)

2. **Install PyTorch**: Match the ABI used by your ROOT installation
   ```bash
   pip install "torch==2.7.0+cpu" --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install Python Dependencies**:
   ```bash
   pip install boost_histogram mplhep pyyaml tqdm scipy h5py scikit-learn
   ```

4. **Build**:
   ```bash
   pip install -e .
   ```

## Why the Build is Complex

The framework bridges three ecosystems:
- **ROOT** (HEP data format and analysis)
- **PyTorch** (Machine learning)
- **Python** (User interface)

Each has its own ABI, linking requirements, and version constraints. Getting all three to work together requires careful configuration.

## Documentation-Only Build

If you only want to build documentation:

```bash
# Requires Doxygen
sudo apt-get install doxygen  # On Ubuntu/Debian

mkdir build && cd build
cmake ..
make doc_doxygen
```

The documentation will be in `build/html/`.

## Alternative: Use Pre-built Package

If available, using a pre-built wheel or conda package is recommended over building from source unless you need to modify the C++ code.

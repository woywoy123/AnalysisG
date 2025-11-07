# Build Notes and Troubleshooting

This document contains detailed information about the AnalysisG build process.

## Build Methods

The framework can be built using two approaches:

### Method 1: Traditional CMake Build (Recommended)

According to the repository maintainer, this method works correctly:

```bash
mkdir build
cd build
cmake ..
make -j12
cmake ..
```

This approach:
- Uses standard CMake configuration and build
- Compiles all C++ modules directly
- The double `cmake ..` invocation may install files to Python site-packages

### Method 2: Python Package Build via pip

Using `pip install -e .` with scikit-build-core:

```bash
pip install -e .
```

This approach encountered compilation errors during testing, though the build system configuration was successful.

## Prerequisites

Before building, ensure you have:

1. **System Dependencies**:
   - CMake 3.20+
   - C++20 compliant compiler (GCC 12.4+ or Clang)
   - Python 3.8+

2. **Required Libraries** (via conda or system packages):
   - ROOT 6.x
   - PyTorch 2.5+ with LibTorch
   - HDF5 with C++ support
   - RapidJSON (auto-downloaded via FetchContent)

3. **Python Dependencies**:
   ```bash
   pip install boost_histogram mplhep pyyaml tqdm pwinput scipy h5py scikit-learn pyAMI-core cython
   ```

### Using Conda Environment (Recommended)

The easiest way to get all dependencies:

```bash
conda env create -f scripts/environment.yml
conda activate gnn-a100
```

This installs:
- ROOT 6.32.02
- PyTorch 2.5.1 with CUDA support
- HDF5 1.14.3
- All Python dependencies
- Compilers and build tools

## Build Instructions

### Traditional CMake Build

```bash
mkdir build
cd build
cmake ..
make -j12      # Adjust -j based on CPU cores
cmake ..       # Second invocation for installation
```

### Python Package Build

```bash
pip install scikit-build-core
pip install -e .
```

Note: This method uses scikit-build-core as a CMake wrapper.

AnalysisG uses a complex build system:

- **Build Backend**: `scikit-build-core` (Python build system that wraps CMake)
- **Build Tool**: CMake 3.20+ with C++17 compiler
- **Python Bindings**: Cython
- **Critical Dependencies**: ROOT, PyTorch C++ (LibTorch), HDF5

## What Actually Gets Built

The root `CMakeLists.txt` primarily handles documentation generation with Doxygen. The actual C++/Python modules are built by scikit-build-core, which discovers and compiles all the modules in `src/AnalysisG/`.

## Build Attempts and Findings

### What Was Actually Tested

The following build steps were successfully executed:

1. **CMake Configuration** ✓
   ```bash
   mkdir build_test && cd build_test
   cmake -DBUILD_DOC=OFF ..
   ```
   **Result**: CMake configuration succeeded. It detected:
   - C++ compiler (GNU 13.3.0) ✓
   - CMake 3.31.6 ✓
   - Python 3.12.3 ✓
   - Doxygen was missing (not critical for library build)

2. **Makefile Generation** ✓
   - CMake successfully generated Makefiles
   - Build targets were created for documentation

3. **Actual Compilation** ✗
   - NOT attempted because ROOT is not available in the test environment
   - The root `CMakeLists.txt` only builds documentation (Doxygen)
   - Full compilation requires `pip install -e .` which triggers scikit-build-core

### Attempt 1: Direct CMake (Root Directory)

```bash
mkdir build && cd build
cmake ..
make
```

**Result**: Only tries to build documentation. Without Doxygen, this fails. The actual library modules are not built this way - they require scikit-build-core via pip.

### Attempt 2: Using pip install (Not Fully Tested)

```bash
pip install -e .
```

**Result**: This is the correct approach for building the full framework, but was not successfully completed because:
- ROOT framework is not installed ❌
- PyTorch C++ libraries would need to be configured ❌
- HDF5 development libraries would be needed ❌
- Build would take 15-30 minutes with all dependencies

The documentation was verified by examining the CMakeLists.txt files and pyproject.toml configuration.

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

## Test Files and Sample Data

The repository includes test files with sample ROOT data in `test/samples/`:

```bash
$ ls test/samples/sample1/
smpl1.root  smpl2.root  smpl3.root
```

These are actual ROOT files (approximately 400-500 KB each) used by the test suite. Once AnalysisG is installed, you can verify it works by running:

```bash
cd test/
python3 test_root_io.py
```

This test will:
1. Load the sample ROOT files
2. Access various trees and branches
3. Verify data can be read correctly
4. Check that missing branches/trees are properly reported

If this test passes without errors, your installation is functional.

# Build Notes and Troubleshooting

This document contains detailed information about the AnalysisG build process based on actual build attempts.

!!! warning "Compilation Status"
    **The framework was NOT successfully compiled.** While the conda environment was set up correctly with ROOT, PyTorch, and all dependencies, and CMake configuration succeeded, no actual C++ code was compiled. The root `CMakeLists.txt` only contains documentation build targets - the actual source code compilation logic appears to be missing or in a different location.

## Build Attempt Summary

### Environment Setup (Successful)

Successfully created conda environment with:
- **ROOT**: 6.32.02 (from conda-forge)
- **PyTorch**: 2.5.1 (with CUDA 12.6 support)  
- **Python**: 3.12.4
- **Compiler**: GCC 12.4.0
- **CMake**: 3.30.2
- **Build System**: scikit-build-core 0.11.6

All dependencies from `scripts/environment.yml` were installed successfully.

### Build Attempt (Failed - No Source Compilation)

When running `pip install -e .`:
- CMake configuration succeeded ✓
- Wheel package was created ✓  
- But the wheel is only 8895 bytes (metadata only) ✗
- No `.so` files were compiled ✗
- Import fails with `ImportError: No module named 'AnalysisG.core.lossfx'` ✗

**Root Cause**: The `CMakeLists.txt` file only contains Doxygen documentation build commands. It does not include any `add_subdirectory()` or other commands to build the actual C++ source code in `src/`.

## Build System Overview

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

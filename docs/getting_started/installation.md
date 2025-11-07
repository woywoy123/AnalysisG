# Installation

This guide provides detailed instructions for installing AnalysisG from source.

## System Requirements

### Prerequisites

Before installing AnalysisG, ensure the following prerequisites are met:

- **C++ Compiler**: GCC 7+ (9.3.0+ recommended) or Clang 5+ (10.0.0+ recommended) with C++17 support
- **CMake**: Version 3.15 or higher
- **Python**: Version 3.8 or higher
- **ROOT**: A working ROOT installation (e.g., `root-framework` via snap or from https://root.cern/)
  - Ensure ROOT environment variables are set: `source /path/to/root/bin/thisroot.sh`
- **HDF5**: Development libraries (e.g., `libhdf5-dev` on Debian/Ubuntu)
- **Boost**: Boost C++ libraries (optional but recommended)

### Automatically Downloaded Dependencies

The following dependencies are automatically downloaded and built by CMake via `FetchContent`:

- **RapidJSON**: For JSON parsing
- **LibTorch (PyTorch C++ API)**: For machine learning functionality

## PyTorch Compatibility

!!! warning "Important: ABI Compatibility"
    AnalysisG requires a specific ABI-compatible version of PyTorch C++. If your Python script uses both the `torch` Python package and the AnalysisG framework, you **must** uninstall any existing `torch` package and install the correct ABI-compatible version.
    
    Failure to do so will likely result in `ImportError: ... undefined symbol: _ZN...` errors due to ABI mismatches between the LibTorch used by AnalysisG and the `torch` Python package.

The framework is currently compiled against **LibTorch 2.7.0**. To ensure compatibility, install the corresponding Python package:

```bash
# For CPU-only support:
pip uninstall torch -y
pip install "torch==2.7.0+cpu" --index-url https://download.pytorch.org/whl/cpu

# For CUDA 12.6 support (ensure your system CUDA matches):
pip uninstall torch -y
pip install "torch==2.7.0+cu126" --index-url https://download.pytorch.org/whl/cu126
```

For other CUDA versions, visit the [PyTorch download page](https://download.pytorch.org/whl/torch_stable.html).

## Installation from Source

!!! warning "Complex Build Process - Critical Dependencies Required"
    Building AnalysisG from source requires **ROOT**, **PyTorch C++ (LibTorch)**, and **HDF5** development libraries. These are not automatically installed. The build will fail without them.
    
    **ROOT is absolutely required** - almost every module links against ROOT libraries. Install ROOT first from https://root.cern/ or use a system where it's already available (e.g., CERN computing infrastructure).

### Prerequisites Check

Before attempting installation, verify you have all required dependencies:

```bash
# Check C++ compiler (must be 7+ for C++17 support)
g++ --version

# Check CMake (must be 3.15+)
cmake --version

# Check Python (must be 3.8+)  
python3 --version

# Check ROOT (CRITICAL - build will fail without this!)
root-config --version
echo $ROOTSYS  # Should show ROOT installation path

# Check if HDF5 development libraries are installed
dpkg -l | grep libhdf5-dev  # On Ubuntu/Debian
# Or: rpm -qa | grep hdf5-devel  # On RHEL/CentOS
```

If ROOT is not installed, see: https://root.cern/install/

### Installation Steps

The framework uses `scikit-build-core` for building, which automatically runs CMake during pip installation.

#### Step 1: Install Python Build Dependencies

```bash
pip install scikit-build-core cython cmake
```

#### Step 2: Install PyTorch (Critical - ABI Must Match)

See the [PyTorch Compatibility](#pytorch-compatibility) section above for version details.

```bash
# Uninstall any existing PyTorch
pip uninstall torch -y

# Install compatible version (CPU)
pip install "torch==2.7.0+cpu" --index-url https://download.pytorch.org/whl/cpu
```

#### Step 3: Install Python Runtime Dependencies

```bash
pip install boost_histogram mplhep pyyaml tqdm pwinput scipy h5py scikit-learn pyAMI-core
```

#### Step 4: Clone and Build

```bash
# Clone the repository
git clone https://github.com/woywoy123/AnalysisG.git
cd AnalysisG

# Build and install (this will take 15-30 minutes)
pip install -e .
```

!!! note "What Happens During Build"
    The `pip install` command triggers scikit-build-core, which:
    
    1. Runs CMake to discover ROOT, PyTorch, HDF5
    2. Compiles all C++ modules in `src/AnalysisG/`
    3. Builds Cython wrappers for Python  
    4. Installs the package to your Python environment
    
    If CMake cannot find ROOT or PyTorch, the build will fail immediately.

### Step 6: Verify Installation

Test that the installation was successful:

```bash
# Test Python import
python3 -c "import AnalysisG; print('AnalysisG imported successfully')"

# Check if core modules are available
python3 -c "from AnalysisG import Analysis; print('Analysis class available')"
```

!!! note "Import Testing"
    If the import fails with symbol errors, verify your PyTorch version matches the ABI used during compilation (see PyTorch Compatibility section above).

## Troubleshooting

### PyTorch ABI Incompatibility

**Symptom**: `ImportError: ... undefined symbol: _ZN...` when importing AnalysisG in Python.

**Solution**: Ensure you have the correct PyTorch version installed:

```bash
python -c "import torch; print(torch.__version__)"
# Should print: 2.7.0+cpu (or cu126)

pip uninstall torch -y
pip install "torch==2.7.0+cpu" --index-url https://download.pytorch.org/whl/cpu
```

### CMake Cannot Find ROOT

**Symptom**: CMake error stating it cannot find ROOT.

**Solution**:

1. Verify ROOT is installed: `root-config --version`
2. Ensure ROOT environment is sourced: `source /path/to/root/bin/thisroot.sh`
3. Check `ROOTSYS` is set: `echo $ROOTSYS`
4. Explicitly tell CMake where ROOT is: `cmake -DROOT_DIR=/path/to/root ..`

### Compilation Errors

**Symptom**: The `make` command fails with compiler errors.

**Solution**:

1. Verify compiler version: `g++ --version` or `clang --version`
2. Ensure all prerequisites are installed
3. Clean and rebuild:
   ```bash
   cd build
   rm -rf *
   cmake ..
   make -j$(nproc)
   ```

### CUDA Version Mismatches

**Symptom**: CUDA-related errors during compilation or runtime.

**Solution**: Ensure your system's CUDA toolkit version (`nvcc --version`) matches the PyTorch CUDA version (e.g., `+cu126` requires CUDA 12.6).

## Next Steps

Once installation is complete, proceed to [First Steps](first_steps.md) to learn how to use AnalysisG.

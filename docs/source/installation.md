# Installation Guide

This guide will walk you through the steps to install AnalysisG and its dependencies.

## Prerequisites

Before installing AnalysisG, ensure you have the following dependencies:

- C++17 compatible compiler (GCC 8+ or Clang 7+)
- CMake 3.14+
- ROOT 6.20+ with PyROOT support
- Python 3.7+
- PyTorch 1.8+
- CUDA toolkit (optional, for GPU acceleration)

## Installation Options

### Option 1: Using conda (Recommended)

The easiest way to install AnalysisG and manage its dependencies is through conda:

```bash
# Create a new conda environment
conda create -n analysisg python=3.9
conda activate analysisg

# Install dependencies
conda install -c conda-forge root cudatoolkit=11.3 pytorch=1.11 -y

# Clone the repository
git clone https://github.com/yourusername/AnalysisG.git
cd AnalysisG

# Build and install
mkdir build && cd build
cmake ..
make -j4
make install
```

### Option 2: Manual Installation

If you prefer to install dependencies manually:

```bash
# Install ROOT (example for Ubuntu)
sudo apt-get install root-system root-plugin-http root-graf-gpad

# Install PyTorch
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Clone the repository
git clone https://github.com/yourusername/AnalysisG.git
cd AnalysisG

# Build and install
mkdir build && cd build
cmake ..
make -j4
make install
```

## Configuration

After installation, set up the environment variables:

```bash
# Add to your .bashrc or equivalent
export ANALYSISG_DIR=/path/to/AnalysisG
export PYTHONPATH=$PYTHONPATH:$ANALYSISG_DIR/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ANALYSISG_DIR/lib
```

## Verification

Verify your installation by running:

```bash
# Test the C++ components
cd $ANALYSISG_DIR/build
ctest -V

# Test the Python bindings
python -c "import AnalysisG; print(AnalysisG.__version__)"
```

## Troubleshooting

If you encounter issues during installation, check the following:

1. **ROOT Version**: Ensure ROOT is properly configured with PyROOT support
   ```bash
   root-config --has-python
   ```

2. **CMake Configuration**: If CMake cannot find dependencies, specify paths manually:
   ```bash
   cmake .. -DROOT_DIR=/path/to/root -DPYTHON_EXECUTABLE=$(which python)
   ```

3. **Library Path Issues**: If you get library loading errors, check LD_LIBRARY_PATH:
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(root-config --libdir)
   ```

For additional help, check the [GitHub issues](https://github.com/yourusername/AnalysisG/issues) or contact the development team.
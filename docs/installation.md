# Installation Guide

## Prerequisites

Before installing AnalysisG, ensure you have the following dependencies:

- C++17 compatible compiler (GCC 8+ or Clang 7+)
- CMake 3.14+
- ROOT 6.20+ with PyROOT support
- Python 3.7+
- PyTorch 1.8+

## Basic Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AnalysisG.git
   cd AnalysisG
   ```

2. Create a build directory and run CMake:
   ```bash
   mkdir build
   cd build
   cmake ..
   ```

3. Compile the project:
   ```bash
   make -j4
   ```

4. Install (optional):
   ```bash
   make install
   ```

## Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Configuration

Configure the environment variables:

```bash
source setup.sh
```

## Testing the Installation

Run the tests to ensure everything is working correctly:

```bash
cd build
ctest
```

## Troubleshooting

If you encounter issues during installation, check the following:

1. Ensure all dependencies are properly installed
2. Check that ROOT is properly configured with PyROOT support
3. Verify that your compiler supports C++17
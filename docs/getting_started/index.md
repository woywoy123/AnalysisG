# Getting Started with AnalysisG

This guide will help you get started with AnalysisG, from installation to running your first analysis.

## Overview

AnalysisG is a framework for particle physics analysis that combines C++ for high-performance computing with Python for accessibility. The framework is designed to:

1. **Process ROOT files** containing particle physics data
2. **Reconstruct physics objects** like jets, leptons, and top quarks
3. **Apply analysis selections** to filter events
4. **Create graphs** for machine learning applications
5. **Train models** using PyTorch-based Graph Neural Networks

## Prerequisites

Before installing AnalysisG, ensure you have the following:

### Required Software

- **C++ Compiler**: GCC 7+ (9.3.0+ recommended) or Clang 5+ (10.0.0+ recommended) with C++17 support
- **CMake**: Version 3.15 or higher
- **Python**: Version 3.8 or higher
- **ROOT**: CERN's ROOT data analysis framework (version 6.20 or higher recommended)

### Optional Dependencies

- **CUDA**: For GPU-accelerated computations (CUDA 12.1 or compatible)
- **HDF5**: For alternative data storage formats

## Next Steps

1. [Installation](installation.md) - Detailed installation instructions
2. [First Steps](first_steps.md) - Your first analysis with AnalysisG

## Need Help?

If you encounter issues during setup, please refer to the [troubleshooting guide](../misc/troubleshooting.rst) in the documentation or open an issue on the GitHub repository.

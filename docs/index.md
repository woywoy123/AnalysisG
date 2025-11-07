# AnalysisG Documentation

Welcome to the AnalysisG framework documentation. AnalysisG is a powerful C++ and Python-based framework designed for high energy physics analysis, particularly for data from the Large Hadron Collider (LHC).

## What is AnalysisG?

AnalysisG provides a comprehensive suite of tools for:

- **Event Reconstruction**: Tools for reconstructing physics objects like jets, leptons, and top quarks from detector-level data
- **Data Analysis**: A flexible framework for implementing analysis selection cuts and producing histograms
- **Graph Neural Networks**: Integration with PyTorch for machine learning applications in particle physics
- **Visualization**: Integrated plotting utilities with support for HEP-standard styles (e.g., ATLAS)

## Key Features

- **C++17 and Python Integration**: High-performance C++ core with convenient Python bindings
- **ROOT Framework Integration**: Seamless integration with CERN's ROOT data analysis framework
- **PyTorch Support**: Built-in support for training Graph Neural Networks (GNNs) on physics data
- **Flexible Template System**: Extensible templates for events, particles, graphs, models, and selections
- **Multi-threaded Processing**: Efficient parallel processing of large datasets

## Quick Links

- [Getting Started](getting_started/index.md) - Installation and first steps
- [User Guide](user_guide/index.md) - Comprehensive usage documentation
- [API Reference](api/index.md) - Detailed API documentation
- [Examples](examples/README.md) - Example analyses and tutorials
- [Development](development/index.md) - Contributing and development guidelines

## System Requirements

- **C++ Compiler**: GCC 7+ or Clang 5+ with C++17 support
- **CMake**: Version 3.15 or higher
- **Python**: Version 3.8 or higher
- **ROOT**: A working ROOT installation
- **PyTorch**: Version 2.7.0 (with specific ABI compatibility)

## License

AnalysisG is open-source software licensed under the MIT License.

## Support

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/woywoy123/AnalysisG).

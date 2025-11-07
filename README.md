# AnalysisG: A Framework for High Energy Physics Analysis

AnalysisG is a C++ and Python-based framework designed for the analysis of high energy physics data, such as that from the Large Hadron Collider (LHC). It provides tools for data processing, event reconstruction, and visualization.

## Project Structure

```
AnalysisG/
├── src/                 # C++ source code for core functionalities
├── python/              # Python bindings and analysis scripts
├── studies/             # Specific physics analysis studies
├── docs/                # Documentation source
├── cmake/               # CMake modules
├── tests/               # Unit and integration tests
└── data/                # Example data or configuration files
```

## Core Concepts

- **Event Reconstruction**: Tools for reconstructing physics objects like jets, leptons, and tops from detector-level data.
- **Data Analysis**: A flexible framework for implementing analysis selection cuts and producing histograms.
- **Visualization**: Integrated plotting utilities to create publication-quality figures.

## Getting Started

To build the project, you can use CMake:
```bash
mkdir build && cd build
cmake ..
make
```

## Documentation

Detailed documentation is available in the `docs/` directory and can be built using Doxygen and Sphinx.

## Requirements

- C++17 compatible compiler
- CMake 3.20+
- ROOT
- Python 3.8+
- Doxygen (for documentation)
- Sphinx (for documentation)

## License

This project is licensed under the MIT License.

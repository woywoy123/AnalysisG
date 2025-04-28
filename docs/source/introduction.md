# Introduction to AnalysisG

AnalysisG is a powerful Graph Neural Network Analysis Framework designed specifically for High Energy Physics applications. It provides a comprehensive suite of tools and modules to represent, analyze, and extract insights from complex physics data using advanced graph neural network techniques.

## Key Features

- **Graph-based Data Representation**: Convert physics events into graph structures where particles become nodes connected by meaningful edges
- **Flexible Neural Network Models**: Implement and train various graph neural network architectures optimized for physics analysis
- **Comprehensive I/O Capabilities**: Seamlessly handle ROOT and other HDF5 formats common in high energy physics
- **Integrated Analysis Pipeline**: Manage the full workflow from data ingestion to model training and evaluation
- **Extensive Metrics & Plotting**: Evaluate model performance with physics-specific metrics and generate publication-quality visualizations
- **Event Selection Tools**: Apply customizable criteria to select and filter physics events of interest

## Framework Architecture

AnalysisG is organized into several core modules, each responsible for specific functionality:

1. **Analysis**: Orchestrates the entire workflow, connecting all other components together
2. **Graph**: Handles graph representation, topology definition, and feature extraction
3. **Model**: Defines machine learning models, training routines, and inference infrastructure
4. **IO**: Manages reading and writing data in various formats (ROOT, HDF5, etc.)
5. **Meta**: Maintains metadata and configuration settings for analysis tasks
6. **Metrics**: Implements performance evaluation and metric tracking
7. **Plotting**: Provides visualization tools for analysis results
8. **Particle**: Defines particle physics objects and their properties
9. **Selection**: Implements event selection algorithms and criteria
10. **Structs**: Offers fundamental data structures used throughout the framework
11. **Tools**: Provides general utility functions for various tasks

## Applications

AnalysisG is particularly well-suited for:

- Top quark tagging and reconstruction
- Jet classification and characterization
- Particle identification in complex environments
- Event classification and anomaly detection
- Feature extraction from raw detector data

## Getting Started

To get started with AnalysisG, follow our [Installation Guide](installation.md) and try the [Quick Start Tutorial](quickstart.md).
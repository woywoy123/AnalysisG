# User Guide Overview

This user guide provides comprehensive information about using the AnalysisG framework for high energy physics analysis with Graph Neural Networks.

## Framework Philosophy

AnalysisG is designed with several key principles in mind:

1. **Modularity**: Each component of the framework is self-contained and can be used independently.
2. **Extensibility**: Users can easily extend the framework with custom components and algorithms.
3. **Efficiency**: The framework is optimized for both development speed and computational performance.
4. **Reproducibility**: Analysis workflows are designed to be fully reproducible.
5. **Physics-awareness**: The framework incorporates domain-specific knowledge from high energy physics.

## Framework Components

The AnalysisG framework consists of several key components that work together to support the full analysis workflow:

### Core Components

- **[Analysis](analysis.md)**: Central orchestrator that coordinates the entire workflow
- **[Graph](graphs.md)**: Representation of physics events as graph structures
- **[Model](models.md)**: Machine learning models for event classification and feature extraction
- **[Metrics](metrics.md)**: Evaluation metrics and performance assessment tools

### Supporting Components

- **IO**: Data input/output handling for various file formats
- **Meta**: Metadata management and configuration
- **Plotting**: Visualization tools for analysis results
- **Particle**: Physics object definitions and operations
- **Selection**: Event selection and filtering mechanisms
- **Tools**: Utility functions and helper classes

## Typical Workflow

A typical analysis workflow with AnalysisG involves the following steps:

1. **Data Preparation**: Load and preprocess physics data (typically from ROOT files)
2. **Graph Construction**: Define how physics events are represented as graphs
3. **Feature Definition**: Specify which physical properties to use as features
4. **Model Configuration**: Set up the machine learning model architecture
5. **Training**: Train the model on the prepared graph data
6. **Evaluation**: Assess model performance using physics-relevant metrics
7. **Inference**: Apply the trained model to new data
8. **Result Visualization**: Generate plots and reports from the analysis

## Configuration System

AnalysisG uses a flexible configuration system based on:

- JSON configuration files for static settings
- C++ settings structures for programmatic configuration
- Command-line parameters for runtime options

## Next Steps

This user guide is organized to help you learn about specific aspects of the framework:

- Learn about [analysis configuration and execution](analysis.md)
- Understand [graph representation of physics events](graphs.md)
- Explore [model architectures and training](models.md)
- Discover [performance metrics and evaluation](metrics.md)

For hands-on examples, please refer to the [Tutorials](../tutorials/index.md) section.
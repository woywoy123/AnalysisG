# AnalysisG Modules Documentation

This directory contains the core modules of the AnalysisG framework, a toolkit for data analysis that appears to be focused on graph-based models, possibly for particle physics data analysis.

## Module Structure

The repository is organized into multiple specialized modules:

- **typecasting**: Handles type conversion between different data representations
- **notification**: Provides notification mechanisms in the framework
- **structs**: Contains data structure definitions used throughout the framework
- **tools**: General utility functions and tools
- **particle**: Classes and functions for particle physics data representation
- **event**: Event-based data handling
- **selection**: Data selection and filtering functionality
- **graph**: Graph construction and manipulation
- **model**: Model definition and training functionality
- **metrics/metric**: Evaluation metrics for model performance
- **meta**: Metadata handling
- **io**: Input/output operations
- **plotting**: Data visualization capabilities
- **lossfx**: Loss functions for model training
- **container**: Data container implementations
- **sampletracer**: Sample tracking functionality
- **analysis**: Analysis workflows
- **dataloader**: Data loading and preprocessing
- **optimizer**: Optimization algorithms for model training

## Key Components

### Model Module

The model module provides templates and implementations for machine learning models. The `model_template` class serves as a base class for implementing different models, with capabilities for:

- Registering modules within the model architecture
- Handling forward passes with graph-based data
- Managing input and output features
- Device management (CPU/GPU)
- Weight initialization

### IO Module

The IO module handles all input/output operations within the framework, including:

- File reading/writing
- Configuration import/export
- Settings management

### Structs Module

This module defines fundamental data structures used throughout the framework:

- Base structures (`bsc_t`)
- Enumerations for data types (`data_enum`)
- Model settings (`model_settings_t`)

## Integration

The modules are integrated through CMake build system, with interdependencies managed through header inclusions. The main `CMakeLists.txt` configures paths and builds all submodules in the correct order of dependencies.

## Key Classes

### model_template

Base class for model implementations with the following key methods:

- `register_module()`: Adds neural network components to the model
- `forward()`: Performs the forward pass for inference
- `assign_features()`: Maps input data to appropriate tensors
- `set_input_features()`: Configures input feature handlers

### io

Handles file operations and settings management:

- `import_settings()`: Loads configuration from external sources

### model_settings_t

Configuration structure for models containing:

- Optimizer settings
- Model naming and device configuration
- Input/output feature mappings
- Inference mode settings

## Development Guidelines

When extending the framework:
1. Add new module directories to `CMakeLists.txt`
2. Ensure proper header inclusion
3. Maintain consistent naming conventions
4. Document public interfaces
5. Follow the existing patterns for object lifecycle management
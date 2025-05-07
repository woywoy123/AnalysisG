# AnalysisG Modules Documentation

This document provides a comprehensive overview of the modules directory within the AnalysisG project.

## Overview

The AnalysisG modules directory contains specialized components that work together to enable advanced data analysis, modeling, and optimization. The modules are organized hierarchically with clear interdependencies as defined in `CMakeLists.txt`.

## Module Structure

As defined in `CMakeLists.txt`, the modules are organized as follows:

- typecasting
- notification
- structs
- tools
- particle
- event
- selection
- graph
- model
- metrics
- metric
- meta
- io
- plotting
- lossfx
- container
- sampletracer
- analysis
- dataloader
- optimizer

## Core Modules

### structs

The `structs` module provides fundamental data structures used throughout the codebase.

#### Key Files:
- `structs/include/structs/base.h`: Defines basic structures and utility functions for type handling
- `structs/include/structs/enums.h`: Contains enumerations used for type identification
- `structs/include/structs/model.h`: Defines structures for model configuration

#### Main Components:

- `bsc_t` struct: Basic structure for handling various data types
  - Provides type translation between ROOT and internal types
  - Handles buffer management for data processing
  - Provides string representation of data types

- `model_settings_t` struct: Configuration settings for models
  - Optimizer settings (`e_optim`, `s_optim`)
  - Model identification (`model_name`, `model_device`)
  - Checkpointing information (`model_checkpoint_path`)
  - Mode settings (`inference_mode`, `is_mc`)
  - Input/output configurations for graph, node, and edge features

- Utility functions:
  - `buildPCM`: Builds PCM files for ROOT dictionary generation
  - `registerInclude`: Registers includes for dictionary generation
  - `buildDict`: Builds dictionary for ROOT integration
  - `buildAll`: Comprehensive build function

### model

The `model` module handles machine learning model creation, training, and inference.

#### Key Components:

- `model_template` class: Base class for all models
  - Constructor initializes property accessors and core components
  - Manages input and output features for graph data
  - Handles module registration and initialization
  - Provides forward pass implementations for both single and batched data
  - Supports cloning for model duplication
  - Handles device management for multi-device training

### analysis

The `analysis` module coordinates high-level analysis workflows.

#### Key Components:

- Core analysis functions:
  - `build_model_session`: Sets up model training environment
  - `progress`: Tracks training/analysis progress
  - `progress_mode`: Reports current processing mode
  - `progress_report`: Generates detailed progress reports
  - `is_complete`: Checks completion status
  - `attach_threads`: Manages thread allocation for parallel processing

### sampletracer

The `sampletracer` module handles sample processing and bookkeeping.

#### Key Components:

- `sampletracer` class:
  - `compile_objects`: Compiles analysis objects across multiple threads
  - Progressive processing with status reporting
  - Thread management for parallel processing

### io

The `io` module manages input/output operations.

#### Key Components:

- `io` class:
  - Constructor/destructor for resource management
  - `import_settings`: Imports configuration parameters

## Feature Modules

### graphs/ssml_mc20

This module provides specialized graph features for particle physics data analysis.

#### Node Features:

- Truth features:
  - `res_node`: Resolution node function
  - `top_node`: Top node function

- Observable features:
  - `pt`: Transverse momentum
  - `eta`: Pseudorapidity
  - `phi`: Azimuthal angle
  - `energy`: Energy measurement
  - `is_lepton`: Lepton identification
  - `is_bquark`: B-quark identification
  - `is_neutrino`: Neutrino identification

## Build System

The CMake build system coordinates the compilation and linking of all modules, with configuration settings generated for both analysis and IO modules.

## Module Interrelationships

- **Data Flow**: Data typically flows from IO through containers and into analysis modules
- **Processing Pipeline**: 
  1. Data loading (IO module)
  2. Data preparation (container/sampletracer)
  3. Model application (model module)
  4. Analysis and optimization (analysis/optimizer)
  5. Result visualization (plotting)

## Development Notes

- The codebase extensively uses C++ templates and ROOT integration
- Threading models are applied for performance-critical operations
- Models follow a standardized interface for consistent integration
- Custom loss functions and metrics can be defined in the lossfx and metrics modules
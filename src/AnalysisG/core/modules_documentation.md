# AnalysisG Modules Documentation

This documentation provides an overview of the modules directory structure in the AnalysisG project. Each module is designed with specific functionality and interacts with other modules to form the complete system.

## Directory Structure

The modules directory contains several subdirectories, each implementing a specific aspect of the system:

```
modules/
├── analysis/
├── container/
├── dataloader/
├── event/
├── graph/
├── io/
├── lossfx/
├── meta/
├── metric/
├── metrics/
├── model/
├── notification/
├── optimizer/
├── particle/
├── plotting/
├── sampletracer/
├── selection/
├── structs/
├── tools/
├── typecasting/
└── CMakeLists.txt
```

## Modules Overview

### structs

The `structs` module provides fundamental data structures and type definitions used throughout the codebase.

#### Key Files:
- `base.cxx`: Implements basic type management, translation between ROOT types and internal data enums
- `include/structs/model.h`: Defines model-related structures like `model_settings_t`
- `include/structs/enums.h`: Contains enum definitions for the project

#### Notable Components:

- `data_enum`: Enumeration for different data types used in the system
- `bsc_t`: Base class for type management with methods like:
  - `root_type_translate`: Translates ROOT string types to internal enum types
  - `as_string`: Converts internal types to string representation
  - `flush_buffer`: Cleans up temporary buffers
- `model_settings_t`: Structure holding model configuration including:
  - Optimizer settings
  - Model name, device, checkpoint path
  - Input/output features for graph, node, and edge components

### model

The `model` module implements the neural network models and related functionality.

#### Key Files:
- `model_template.cxx`: Implements the base template for models in the system

#### Notable Components:

- `model_template` class: Base class for all models with methods like:
  - Constructor: Sets up input/output features and device properties
  - `clone()`: Creates a copy of the model
  - `register_module()`: Registers torch modules with the model
  - `forward()`: Processes input graphs (different overloads for single/multiple graphs)
  - `assign_features()`: Maps input features to tensors
  - `set_input_features()`: Configures input feature mappings

### event

The `event` module handles event data management and processing.

#### Key Files:
- `event_template.cxx`: Implements base template for event handling

#### Notable Components:

- `event_template` class: Base class for event handling with properties:
  - `trees`: Tree collection management
  - `branches`: Branch management within trees
  - `leaves`: Access to leaf values
  - `name`: Event name handling
  - `hash`: Event hash identification
  - `tree`: Current tree accessor
  - `weight`: Event weight handling
  - `index`: Event indexing

### io

The `io` module handles input/output operations across the system.

#### Key Files:
- `io.cxx`: Implements core IO functionality

#### Notable Components:

- `io` class: Handles file operations with methods like:
  - Constructor/Destructor: Setup and cleanup of IO resources
  - `import_settings()`: Imports configuration settings

### sampletracer

The `sampletracer` module tracks and manages samples throughout processing.

#### Key Files:
- `sampletracer.cxx`: Implements sample tracking functionality

#### Notable Components:

- `sampletracer` class: Manages sample processing with methods like:
  - `compile_objects()`: Organizes and processes objects with multi-threading support
  - Support for progress tracking and visualization
  - Thread management for parallel processing

## Module Interactions

The modules interact in a structured manner:

1. **structs** provides base data types used by all other modules
2. **io** handles data input/output used by most processing modules
3. **event** manages event data that flows into models
4. **model** implements neural network functionality using data from events
5. **sampletracer** coordinates the processing of multiple samples

## Build System

The build system is managed through CMake with the main configuration in `CMakeLists.txt`. The modules are compiled in a specific order to respect dependencies:

1. Core utilities (typecasting, notification, structs, tools)
2. Physics objects (particle, event, selection)
3. ML components (graph, model, metrics, metric)
4. Infrastructure (meta, io, plotting, lossfx)
5. Application level (container, sampletracer, analysis, dataloader, optimizer)

## Type System

The codebase implements a custom type system that bridges between ROOT data types and internal representations:

- ROOT types like `Float_t`, `Double_t`, `Int_t` are mapped to internal enum values
- Vector types are detected and handled separately
- Custom string representation for debugging and serialization
# AnalysisG Modules Documentation

This directory contains the core modules of the AnalysisG framework. Each module encapsulates specific functionality that together forms the complete analysis system.

## Directory Structure

```
src/modules/
├── structs/            # Data structures and models
│   ├── cxx/            # C++ implementations 
│   └── include/        # Header files
│       └── structs/    # Structure-specific headers
```

## Module: structs

The `structs` module provides fundamental data structures and models used throughout the AnalysisG framework.

### Key Components

#### Base (base.h/base.cxx)

The base component provides core functionality and utilities:

- `bsc_t`: Base class with string manipulation and data conversion utilities
  - `root_type_translate()`: Converts string representations to data types (enum)
  - `as_string()`: String representation functionality
  - `scan_buffer()`: Buffer scanning capability
  - `flush_buffer()`: Clears internal buffers

#### Support Functions

- `buildDict(std::string _name, std::string _shrt)`: Builds dictionary mappings
- `registerInclude(std::string pth, bool is_abs)`: Registers include paths
- `buildPCM(std::string name, std::string incl, bool exl)`: Builds precompiled modules
- `buildAll()`: Orchestrates the build process for all components
- `count(const std::string* str, const std::string sub)`: String counting utility

#### Models (model.h)

Models define data structures for machine learning components:

- `model_settings_t`: Configuration struct for model parameters including:
  - Optimization settings (`e_optim`, `s_optim`)
  - Naming conventions (`weight_name`, `tree_name`, `model_name`)
  - Device configuration (`model_device`)
  - Path management (`model_checkpoint_path`)
  - Operating modes (`inference_mode`, `is_mc`)
  - Graph representation configuration:
    - Output mappings (`o_graph`, `o_node`, `o_edge`)
    - Input definitions (`i_graph`, `i_node`, `i_edge`)

#### Enumerations (enums.h)

Contains enumeration types used throughout the system:

- `data_enum`: Data type classifications
- `opt_enum`: Optimization strategy options

## Development Guidelines

When extending the codebase:

1. For adding new types, refer to the "Add your type (2)" section in base.cxx
2. For adding routing functionality, use the "(3). add the routing" section
3. For buffer management, follow the "(4.) Add the buffer flush" pattern
4. Keep consistent naming conventions with existing code

## Dependencies

The module relies on:
- C++ Standard Library
- ROOT Framework (TInterpreter)
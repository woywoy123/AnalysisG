C++ Modules Documentation
=========================

This directory contains complete documentation for all C++ backend modules in AnalysisG.

Documentation Structure
-----------------------

The modules are organized into categories:

**Core Template Implementations**
- `analysis.rst` - Analysis execution engine
- `event.rst` - Event template (C++ implementation)
- `particle.rst` - Particle template (C++ implementation)
- `graph.rst` - Graph template for GNNs
- `selection.rst` - Selection template for event filtering

**Analysis Components**  
- `optimizer.rst` - Training optimizers
- `lossfx.rst` - Loss functions and optimizer configuration

**Specialized Modules**
- `metrics.rst` - Evaluation metrics
- `nusol.rst` - Neutrino reconstruction
- `plotting.rst` - Plotting utilities

**Infrastructure** (to be documented)
- I/O operations
- Metadata handling
- Data structures
- Utility functions
- Type conversion

Key Features Documented
-----------------------

Each module documentation includes:

1. **C++ Class Definitions** - Complete class hierarchies with cproperty accessors
2. **Method Signatures** - Full parameter lists and return types
3. **Usage Examples** - Both C++ and Python integration examples
4. **Mathematical Formulations** - Physics equations and algorithms
5. **Implementation Details** - File locations and structure
6. **Integration Patterns** - How modules work together

C++ Property System
-------------------

AnalysisG uses a `cproperty` template system for properties:

```cpp
cproperty<double, particle_template> pt;
```

This generates:
- Automatic setter: `set_pt(double*, particle_template*)`
- Automatic getter: `get_pt(double*, particle_template*)`
- Python-accessible property through Cython wrappers

Template Pattern
----------------

All template classes (event, particle, graph, selection, metric, model) follow a consistent pattern:

1. **C++ Base Class** - In `src/AnalysisG/modules/<type>/`
   - Virtual methods for customization
   - cproperty accessors for configuration
   - Core implementation

2. **Cython Wrapper** - In `src/AnalysisG/core/<type>_template.pyx`
   - Python-accessible interface
   - Pointer management
   - Type conversions

3. **User Subclasses** - In `src/AnalysisG/<types>/`
   - Override virtual methods
   - Implement analysis-specific logic
   - Register with Analysis

Building and Integration
------------------------

C++ modules are compiled with CMake and linked with:
- ROOT (for I/O)
- LibTorch (for ML operations)
- CUDA (for GPU acceleration in PyC)

Python integration happens through Cython wrappers that:
- Expose C++ classes to Python
- Handle memory management
- Provide Pythonic interfaces

For more information, see the main README and installation guide.

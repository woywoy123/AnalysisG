# Cython Python Bindings Documentation

## Overview

AnalysisG uses **Cython** to create Python bindings for C++ classes. These bindings allow Python users to access the high-performance C++ framework through a Pythonic interface.

## File Types

### .pyx Files (Python Implementation)
- **Count**: 53 files
- **Purpose**: Cython implementation files containing Python wrapper code
- **Location**: `src/AnalysisG/core/*.pyx`, `src/AnalysisG/*/**.pyx`
- **Content**: `cdef class` wrappers around C++ classes

**Example** (`graph_template.pyx`):
```cython
cdef class GraphTemplate:
    def __cinit__(self):
        self.ptr = new graph_template()
    
    def __dealloc__(self):
        del self.ptr
    
    @property
    def index(self):
        return self.ptr.index
```

### .pxd Files (Cython Headers)
- **Count**: 100 files
- **Purpose**: Cython declaration files (similar to C++ .h files)
- **Location**: `src/AnalysisG/**/__init__.pxd`, `src/AnalysisG/*/*.pxd`
- **Content**: External C++ declarations and Cython class definitions

**Example** (`graph_template.pxd`):
```cython
from libcpp cimport bool
from AnalysisG.modules.graph cimport graph_template

cdef class GraphTemplate:
    cdef graph_template* ptr
```

## Doxygen Integration

### Automatic Processing

The `Doxyfile` configuration automatically processes Cython files:

```ini
# File patterns include Cython
FILE_PATTERNS = *.h *.cxx *.cu *.cuh *.pyx *.pxd *.dox

# Map Cython to C++ for parsing
EXTENSION_MAPPING = pyx=C++ pxd=C++
```

### How It Works

1. **Parsing**: Doxygen treats .pyx/.pxd as C++ files
2. **Extraction**: Class definitions and docstrings extracted
3. **Documentation**: Merged with C++ documentation
4. **Output**: Unified API reference in HTML

### Why No Separate .dox Files?

**Cython files don't need separate documentation because:**
- They are thin wrappers around C++ classes
- C++ documentation already covers the functionality
- Doxygen automatically extracts Python-visible API
- Docstrings in .pyx files are included in output

## Documentation Strategy

### For C++ Core (Documented)
- **Files**: `src/AnalysisG/modules/*/include/**/*.h`
- **Documentation**: Comprehensive .dox files (28 modules)
- **Purpose**: Detailed API reference, usage examples, workflows

### For Cython Bindings (Auto-processed)
- **Files**: `src/AnalysisG/core/*.pyx`, `src/AnalysisG/*/*.pyx`
- **Documentation**: Extracted automatically via EXTENSION_MAPPING
- **Purpose**: Python API surface for C++ functionality

## File Statistics

```
Cython Python Bindings:
├── .pyx files:     53 (Python implementations)
├── .pxd files:    100 (Cython declarations)
└── Total:         153 Cython files

Coverage:
├── Core wrappers:   14 files (graph_template, event_template, etc.)
├── Selections:      19 files (mc16, mc20, analysis, etc.)
├── Models:           3 files (GNN models)
├── Events:          11 files (particle/event implementations)
└── Utilities:        9 files (tools, notification, plotting, etc.)
```

## Key Cython Classes

### Core Templates (src/AnalysisG/core/)
- `GraphTemplate` - Graph construction wrapper
- `EventTemplate` - Event container wrapper
- `ParticleTemplate` - Particle object wrapper
- `SelectionTemplate` - Event selection wrapper
- `ModelTemplate` - GNN model wrapper

### Analysis Tools
- `Meta` - Metadata management
- `Notification` - Logging and progress bars
- `ROC` - ROC curve analysis
- `OptimizerConfig` - Training configuration

### Implementations
- `event_*.pyx` - Physics event implementations
- `particle_*.pyx` - Particle type implementations
- `selection_*.pyx` - Physics selection implementations
- `graph_*.pyx` - Graph construction implementations

## Accessing Documentation

### In Generated Doxygen Docs

1. **Navigate to Classes**:
   - HTML output includes Cython classes
   - Listed alongside C++ classes
   - Marked with namespace/module info

2. **Search Functionality**:
   - Search for class names (e.g., "GraphTemplate")
   - Find both C++ and Python interfaces
   - Cross-references between implementations

3. **Source Browser**:
   - Browse .pyx source code
   - Syntax highlighting applied
   - References to C++ implementation

### In Python (Runtime)

```python
import AnalysisG

# Access docstrings
help(AnalysisG.GraphTemplate)
print(AnalysisG.EventTemplate.__doc__)

# Introspection
dir(AnalysisG.Meta)
```

## Validation

The `build_docs.sh` script validates Cython integration:

```bash
./build_docs.sh --validate-only
```

**Checks performed:**
- ✓ FILE_PATTERNS includes *.pyx *.pxd
- ✓ EXTENSION_MAPPING configured for Cython
- ✓ Counts .pyx and .pxd files
- ✓ Verifies Doxygen can parse Cython syntax

## Example: GraphTemplate Documentation Flow

### 1. C++ Core Implementation
```cpp
// src/AnalysisG/modules/graph/include/templates/graph_template.h
class graph_template : public tools {
public:
    void CompileEvent();
    bool add_node_data_feature(std::string var);
    // ... (fully documented in C++)
};
```

### 2. Cython Wrapper
```cython
# src/AnalysisG/core/graph_template.pyx
cdef class GraphTemplate:
    """Python wrapper for graph_template C++ class."""
    cdef graph_template* ptr
    
    def CompileEvent(self):
        """Compile event into graph structure."""
        self.ptr.CompileEvent()
```

### 3. Doxygen Processing
- Reads C++ header → Generates detailed C++ docs
- Reads .pyx file → Extracts Python API
- Merges both → Unified documentation showing:
  - C++ implementation details
  - Python interface
  - Cross-references

### 4. Output
Users see both:
- **C++ API**: Full implementation details, templates, overloads
- **Python API**: Pythonic interface, property accessors, docstrings

## Best Practices

### When to Add Docstrings

**In .pyx files, add docstrings for:**
- Public methods visible to Python users
- Complex property accessors
- Wrapper-specific behavior

**Example:**
```cython
def add_selection(self, SelectionTemplate sel):
    """
    Add a selection instance to the analysis.
    
    Parameters:
        sel (SelectionTemplate): Selection to apply to events
    
    Returns:
        bool: True if selection added successfully
    """
    return self.ptr.add_selection(sel.ptr)
```

### What Not to Document

**Don't duplicate C++ documentation for:**
- Simple property wrappers (auto-documented)
- Direct C++ method calls (refer to C++)
- Internal Cython implementation details

## Summary

- **153 Cython files** automatically processed by Doxygen
- **No separate .dox files needed** for Cython bindings
- **EXTENSION_MAPPING** handles Cython→C++ translation
- **Unified documentation** shows both C++ and Python APIs
- **build_docs.sh** validates Cython configuration

The Cython files are **not missing** - they are **automatically integrated** into the documentation system through Doxygen's extension mapping feature.

---

**See Also:**
- [Doxygen Configuration](../../Doxyfile) - EXTENSION_MAPPING settings
- [Build Script](build_docs.sh) - Validation for Cython files
- [README](README.md) - Complete documentation overview

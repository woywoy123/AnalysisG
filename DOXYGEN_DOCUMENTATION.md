# AnalysisG API Documentation with Doxygen

This document provides comprehensive information about generating and using API documentation for the AnalysisG project using Doxygen.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [What Gets Documented](#what-gets-documented)
5. [Documentation Structure](#documentation-structure)
6. [Customizing the Documentation](#customizing-the-documentation)
7. [Understanding the Generated Documentation](#understanding-the-generated-documentation)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)
10. [Contributing Documentation](#contributing-documentation)

## Overview

AnalysisG uses [Doxygen](https://www.doxygen.nl/) to automatically generate comprehensive API documentation from the C++ and CUDA source code. The documentation includes:

- **Class hierarchies and inheritance diagrams** - Visual representation of object-oriented design
- **Include dependency graphs** - Shows which files depend on which headers
- **Function call relationships** - Traces code execution flow
- **Source code browser** - Syntax-highlighted, cross-referenced source code
- **API reference** - Complete documentation of classes, functions, variables, and types

### Why Doxygen?

- **Automated**: Documentation is generated directly from source code
- **Always Current**: Regenerate anytime to reflect latest code changes
- **Cross-Referenced**: Click any identifier to jump to its definition
- **Visual**: Diagrams help understand complex relationships
- **Standard**: Industry-standard tool used by many major projects

## Prerequisites

### Required Software

1. **Doxygen** (version 1.8.0 or higher)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install doxygen
   
   # macOS with Homebrew
   brew install doxygen
   
   # From source
   # See: https://www.doxygen.nl/download.html
   ```

2. **Graphviz** (for diagram generation)
   ```bash
   # Ubuntu/Debian
   sudo apt-get install graphviz
   
   # macOS with Homebrew
   brew install graphviz
   ```

### Verification

Verify installation:
```bash
doxygen --version  # Should show 1.8.0 or higher
dot -V             # Should show Graphviz version
```

## Quick Start

### Step 1: Generate Documentation

From the repository root directory:
```bash
doxygen Doxyfile
```

This will create a `doxygen-docs/` directory containing all generated documentation.

**Expected Output:**
```
Searching for include files...
Searching for files to process...
Reading and parsing tag files
Parsing files
Building class list...
Building namespace list...
[...]
Generating HTML output...
finished...
```

**Generation Time:** Approximately 2-5 minutes depending on your system.

### Step 2: View Documentation

Open the main documentation page in your web browser:

```bash
# Linux
firefox doxygen-docs/html/index.html

# macOS
open doxygen-docs/html/index.html

# Windows
start doxygen-docs/html/index.html

# Or use a local web server (recommended for testing)
cd doxygen-docs/html
python3 -m http.server 8000
# Then open http://localhost:8000 in your browser
```

## What Gets Documented

### Source Files

The documentation covers **437 source files** in `src/AnalysisG/`:

#### C++ and CUDA Files (284 files)

- **14 CUDA source files** (.cu) - GPU kernel implementations
- **15 CUDA header files** (.cuh) - CUDA kernel declarations
- **147 C++ implementation files** (.cxx) - Core algorithm implementations
- **108 C++ header files** (.h) - Class and function declarations

#### Cython Files (153 files)

- **Cython implementation files** (.pyx) - Python-C++ interface implementations
  - Event and particle wrappers
  - Metric interfaces
  - Model interfaces
  - Template base class wrappers
  - PyC computation kernel wrappers

- **Cython declaration files** (.pxd) - Cython header equivalents
  - C++ class declarations for Cython
  - Type definitions
  - External function signatures
  - Import declarations

**Total Source Lines:** Over 100,000 lines of documented code

| File Type | Extension | Count | Description |
|-----------|-----------|-------|-------------|
| C++ Headers | `.h` | 108 | Class and function declarations |
| C++ Implementation | `.cxx` | 147 | Function implementations |
| CUDA Source | `.cu` | 14 | CUDA kernel implementations |
| CUDA Headers | `.cuh` | 15 | CUDA kernel declarations |
| Cython Implementation | `.pyx` | ~76 | Python-C++ interface implementations |
| Cython Headers | `.pxd` | ~77 | Cython declarations and type definitions |
| **Total** | - | **437** | **All source files** |

### Modules Documented

The following major modules are included:

#### 1. **Metrics** (`src/AnalysisG/metrics/`)
- `accuracy/` - Accuracy calculation metrics (C++ + Cython)
- `pagerank/` - PageRank algorithm implementation (C++ + Cython)

#### 2. **Events** (`src/AnalysisG/events/`)
- `exp_mc20/` - MC20 experimental event definitions (C++ + Cython)
- `bsm_4tops/` - Beyond Standard Model 4-tops events (C++ + Cython)
- `gnn/` - Graph Neural Network event structures (C++ + Cython)
- `ssml_mc20/` - SSML MC20 event definitions (C++ + Cython)

#### 3. **Models** (`src/AnalysisG/models/`)
- `grift/` - GRIFT model implementation (C++ + Cython)
- `RecursiveGraphNeuralNetwork/` - Recursive GNN architecture (C++ + Cython)

#### 4. **Selections** (`src/AnalysisG/selections/`)
- Event selection algorithms for various analyses (C++ + Cython)
- Truth matching and kinematic selections
- Region definitions

#### 5. **Templates** (`src/AnalysisG/modules/`)
- `EventTemplate` - Base class for events (C++ + Cython)
- `ParticleTemplate` - Base class for particles (C++ + Cython)
- `GraphTemplate` - Base class for graphs (C++ + Cython)
- `SelectionTemplate` - Base class for selections (C++ + Cython)
- `ModelTemplate` - Base class for models (C++ + Cython)

#### 6. **PyC (Python-CUDA Interface)** (`src/AnalysisG/pyc/`)
- `operators/` - Tensor operations (C++/CUDA + Cython)
- `nusol/` - Neutrino solution algorithms (C++/CUDA + Cython)
- `physics/` - Physics calculations (ΔR, invariant mass, etc.) (C++/CUDA + Cython)
- `transform/` - Coordinate transformations (C++/CUDA + Cython)
- `graph/` - Graph operations (C++/CUDA + Cython)
- `operators/` - Tensor operations
- `nusol/` - Neutrino solution algorithms
- `physics/` - Physics calculations (ΔR, invariant mass, etc.)
- `transform/` - Coordinate transformations
- `graph/` - Graph operations
- `cutils/` - C++ utility functions

### Excluded from Documentation

The following are intentionally excluded:

- Test files (`*/test/*`, `*/tests/*`)
- Build artifacts (`*/build/*`)
- Version control files (`*/.git/*`)
- Python bindings (`.pyx`, `.pxd` files)
- Compiled outputs (`.so`, `.o` files)

## Documentation Structure

### HTML Output Organization

```
doxygen-docs/
└── html/
    ├── index.html              # Main documentation page (start here)
    ├── classes.html            # List of all classes
    ├── files.html              # List of all files
    ├── namespaces.html         # List of namespaces
    ├── hierarchy.html          # Class hierarchy
    ├── functions.html          # Function index
    ├── dir_*.html              # Directory documentation
    ├── class*.html             # Individual class pages
    ├── *.png                   # Generated diagrams (~835 images)
    ├── search/                 # Search functionality
    └── *.css, *.js             # Styling and navigation
```

### Navigation Guide

1. **Main Page** (`index.html`)
   - Project overview
   - Quick links to major sections
   - Getting started guide

2. **Classes Tab**
   - Alphabetical list of all classes
   - Click any class to see:
     - Inheritance diagram
     - Member functions and variables
     - Detailed descriptions
     - Source code

3. **Files Tab**
   - Directory structure
   - File list with descriptions
   - Include dependency graphs
   - Source code browser

4. **Search Box** (top-right)
   - Full-text search across all documentation
   - Search by class name, function name, or keyword
   - Real-time search results

## Customizing the Documentation

### Modifying the Doxyfile

The `Doxyfile` contains extensive inline documentation for each configuration option. Key sections:

#### Project Information
```
PROJECT_NAME    = "AnalysisG"
PROJECT_BRIEF   = "A Graph Neural Network Analysis Framework for High Energy Physics"
PROJECT_NUMBER  = "1.0"  # Update this for new versions
```

#### Input Configuration
```
INPUT           = src/AnalysisG
FILE_PATTERNS   = *.h *.cxx *.cu *.cuh
EXCLUDE_PATTERNS = */test/* */tests/* */.git/* */build/*
```

#### Output Location
```
OUTPUT_DIRECTORY = doxygen-docs
HTML_OUTPUT      = html
```

### Adding New File Types

To document additional file types (e.g., `.cpp`, `.hpp`):

1. Edit `Doxyfile`
2. Find `FILE_PATTERNS`
3. Add new extensions: `FILE_PATTERNS = *.h *.cxx *.cu *.cuh *.cpp *.hpp`
4. Regenerate: `doxygen Doxyfile`

### Changing Color Scheme

Modify HTML appearance in `Doxyfile`:

```
HTML_COLORSTYLE_HUE    = 220  # 0-359: 0=red, 120=green, 220=blue, 240=purple
HTML_COLORSTYLE_SAT    = 100  # 0-255: saturation level
HTML_COLORSTYLE_GAMMA  = 80   # 40-240: brightness
```

### Enabling/Disabling Diagrams

Control diagram generation in `Doxyfile`:

```
HAVE_DOT            = YES  # Set to NO to disable all diagrams
CLASS_GRAPH         = YES  # Class inheritance diagrams
INCLUDE_GRAPH       = YES  # File dependency graphs
COLLABORATION_GRAPH = YES  # Class usage diagrams
CALL_GRAPH          = NO   # Function call graphs (can be very large)
```

## Understanding the Generated Documentation

### Class Documentation Pages

Each class page includes:

1. **Inheritance Diagram**
   - Shows base classes (above) and derived classes (below)
   - Color coding: blue=documented, grey=external

2. **Collaboration Diagram**
   - Shows member relationships and composition
   - Helps understand class dependencies

3. **Member Lists**
   - Public/Protected/Private sections
   - Organized by type (constructors, methods, variables)

4. **Detailed Descriptions**
   - Extracted from source code comments
   - Parameter descriptions
   - Return value information

5. **Source Code Links**
   - Click "Go to source code" to see implementation
   - Syntax highlighting with cross-references

### File Documentation Pages

Each file page includes:

1. **Include Graph**
   - Shows all headers included by this file
   - Transitive dependencies visualized

2. **Included By Graph**
   - Shows which files include this header
   - Impact analysis for header changes

3. **Function/Variable Lists**
   - All definitions in the file
   - Links to detailed documentation

4. **Source Browser**
   - Complete file contents
   - Click any identifier to jump to definition

### Directory Documentation

Shows:
- Subdirectory structure
- Files in each directory
- Directory dependency graphs
- Helps understand modular organization

## Troubleshooting

### Common Issues and Solutions

#### Issue: "doxygen: command not found"
**Solution:** Install Doxygen
```bash
sudo apt-get install doxygen
```

#### Issue: "dot: command not found" or missing diagrams
**Solution:** Install Graphviz
```bash
sudo apt-get install graphviz
```

#### Issue: Warning "could not open file for reading"
**Cause:** File permissions or path issues
**Solution:** 
```bash
# Check file permissions
ls -la src/AnalysisG/

# Ensure you're in the repository root
pwd  # Should show .../AnalysisG
```

#### Issue: Documentation is incomplete or missing classes
**Possible Causes:**
1. Files not matching `FILE_PATTERNS` in Doxyfile
2. Files in excluded directories
3. Parsing errors

**Solution:**
```bash
# Run doxygen with full output to see warnings
doxygen Doxyfile 2>&1 | tee doxygen.log

# Check log for parsing errors or excluded files
grep -i "warning\|error" doxygen.log
```

#### Issue: Diagrams are missing or broken
**Solution:**
```bash
# Verify Graphviz is working
dot -V

# Check Doxyfile settings
grep HAVE_DOT Doxyfile  # Should be YES
grep DOT_PATH Doxyfile   # Should be empty or point to dot executable
```

#### Issue: Documentation generation is very slow
**Optimization Options:**
1. Disable call graphs (already disabled by default)
2. Reduce graph complexity:
   ```
   DOT_GRAPH_MAX_NODES = 30  # Reduce from 50
   ```
3. Disable some diagram types:
   ```
   COLLABORATION_GRAPH = NO
   DIRECTORY_GRAPH = NO
   ```

### Getting Help

If you encounter issues not covered here:

1. **Check Doxygen Manual:** https://www.doxygen.nl/manual/
2. **Review Doxyfile Comments:** The file contains extensive inline documentation
3. **Enable Verbose Output:** Set `QUIET = NO` in Doxyfile
4. **Check Log Files:** Review warnings and errors during generation

## Advanced Usage

### Generating Documentation for Specific Modules

To document only specific subdirectories:

1. Create a custom Doxyfile (e.g., `Doxyfile.metrics`)
2. Modify `INPUT`:
   ```
   INPUT = src/AnalysisG/metrics
   ```
3. Generate: `doxygen Doxyfile.metrics`

### Integrating with CI/CD

Add documentation generation to your CI pipeline:

```yaml
# Example for GitHub Actions
- name: Generate Documentation
  run: |
    sudo apt-get install -y doxygen graphviz
    doxygen Doxyfile
    
- name: Deploy to GitHub Pages
  uses: peaceiris/actions-gh-pages@v3
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    publish_dir: ./doxygen-docs/html
```

### Creating PDF Documentation

Enable LaTeX output in Doxyfile:

```
GENERATE_LATEX = YES
```

Then compile to PDF:

```bash
doxygen Doxyfile
cd doxygen-docs/latex
make pdf
```

### Extracting Specific Information

Doxygen can generate XML output for custom processing:

```
GENERATE_XML = YES
```

Use the XML output for:
- Custom documentation formats
- Automated quality checks
- Integration with other tools

### Setting Up Auto-Regeneration

Use `inotify` (Linux) to auto-regenerate on file changes:

```bash
# Install inotify-tools
sudo apt-get install inotify-tools

# Watch for changes and regenerate
while inotifywait -r -e modify src/AnalysisG/; do
    doxygen Doxyfile
done
```

## Contributing Documentation

### Writing Doxygen Comments

Doxygen extracts documentation from specially formatted comments in source code.

#### Basic Comment Styles

**Option 1: JavaDoc Style (Recommended)**
```cpp
/**
 * @brief Calculates the invariant mass of two particles
 * 
 * @param p1 Four-momentum of first particle
 * @param p2 Four-momentum of second particle
 * @return double The invariant mass in GeV
 */
double invariantMass(const FourVector& p1, const FourVector& p2);
```

**Option 2: Qt Style**
```cpp
/*!
 * \brief Calculates the invariant mass of two particles
 * 
 * \param p1 Four-momentum of first particle
 * \param p2 Four-momentum of second particle
 * \return double The invariant mass in GeV
 */
double invariantMass(const FourVector& p1, const FourVector& p2);
```

**Option 3: Single-Line**
```cpp
/// @brief Calculates the invariant mass of two particles
double invariantMass(const FourVector& p1, const FourVector& p2);
```

#### Class Documentation

```cpp
/**
 * @class ParticleTemplate
 * @brief Base template class for particle objects in AnalysisG
 * 
 * This class provides a polymorphic interface for particle objects,
 * allowing different particle types (electrons, muons, jets) to be
 * treated uniformly in event processing.
 * 
 * @details
 * The ParticleTemplate supports:
 * - Four-momentum calculations
 * - Truth matching
 * - Parent-child relationships
 * 
 * @see EventTemplate, GraphTemplate
 */
class ParticleTemplate {
    // ...
};
```

#### File Documentation

Add to the top of each file:

```cpp
/**
 * @file particles.h
 * @brief Particle template class definitions
 * @author AnalysisG Team
 * @date 2024
 * 
 * This file contains the template classes used for representing
 * particles in high energy physics events.
 */
```

#### Common Doxygen Tags

| Tag | Purpose | Example |
|-----|---------|---------|
| `@brief` | Short description | `@brief Calculates DeltaR` |
| `@param` | Function parameter | `@param eta Pseudorapidity` |
| `@return` | Return value | `@return Distance in eta-phi space` |
| `@see` | Cross-reference | `@see calculatePhi()` |
| `@note` | Important note | `@note This function is GPU-accelerated` |
| `@warning` | Warning message | `@warning Not thread-safe` |
| `@deprecated` | Deprecated feature | `@deprecated Use calculateDeltaR2() instead` |
| `@todo` | Future work | `@todo Add support for massive particles` |

### Documentation Quality Guidelines

1. **Write Brief Descriptions**
   - One sentence summarizing the purpose
   - Start with a verb (calculates, performs, returns)

2. **Document All Parameters**
   - Explain meaning, units, and valid ranges
   - Note special values (e.g., nullptr, -1)

3. **Explain Return Values**
   - Describe what is returned
   - Document error conditions

4. **Include Examples**
   ```cpp
   /**
    * @brief Calculates Delta R between two particles
    * 
    * @code
    * Particle p1, p2;
    * double dr = calculateDeltaR(p1.eta(), p1.phi(), 
    *                              p2.eta(), p2.phi());
    * if (dr < 0.4) {
    *     // Particles are close
    * }
    * @endcode
    */
   ```

5. **Cross-Reference Related Code**
   - Link to related classes and functions
   - Help users navigate the codebase

### Verifying Your Documentation

After adding documentation:

1. Regenerate: `doxygen Doxyfile`
2. Check for warnings: `doxygen Doxyfile 2>&1 | grep -i warning`
3. Review in browser: Open your class/file in HTML output
4. Verify:
   - All parameters documented
   - Return value explained
   - Examples compile
   - Cross-references work

## Documentation Statistics

Current documentation coverage (as of last generation):

```
Total Source Files:        284
- C++ Headers (.h):        108
- C++ Source (.cxx):       147  
- CUDA Source (.cu):       14
- CUDA Headers (.cuh):     15

Generated Output:
- HTML Pages:              ~1,665
- Diagrams (PNG):          ~835
- Total Size:              ~133 MB

Generation Time:           ~2-5 minutes
```

## Additional Resources

### Official Documentation
- **Doxygen Manual:** https://www.doxygen.nl/manual/
- **Configuration Reference:** https://www.doxygen.nl/manual/config.html
- **Special Commands:** https://www.doxygen.nl/manual/commands.html

### Examples and Tutorials
- **Doxygen Examples:** https://www.doxygen.nl/manual/examples.html
- **Comment Styles:** https://www.doxygen.nl/manual/docblocks.html

### Related Tools
- **Graphviz:** https://graphviz.org/
- **CMake Integration:** https://cmake.org/cmake/help/latest/module/FindDoxygen.html

## Support and Feedback

For issues specific to AnalysisG documentation:
- Open an issue on GitHub
- Include relevant sections of `doxygen.log`
- Describe expected vs actual behavior

For Doxygen-specific questions:
- Check the official manual first
- Search Stack Overflow for similar issues
- Consider the Doxygen mailing list

---

**Last Updated:** 2024  
**Doxygen Version:** 1.9.8+  
**AnalysisG Version:** 1.0

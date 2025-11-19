# AnalysisG Doxygen Documentation

This directory contains comprehensive Doxygen documentation for the AnalysisG framework.

## Documentation Structure

```
docs/doxygen/
├── overview.dox              # Framework overview and pipeline workflow
├── modules_index.dox         # Complete module index (NEW)
├── module_interactions.dox   # Module dependency graphs
├── modules/                  # Core module documentation
│   ├── analysis/
│   ├── container/            # (NEW) File-level data container
│   ├── dataloader/
│   ├── graph/
│   ├── io/
│   ├── lossfx/
│   ├── meta/
│   ├── metric/
│   ├── metrics/              # (NEW) Training analytics & visualization
│   ├── model/
│   ├── notification/
│   ├── optimizer/
│   ├── plotting/             # (NEW) ROOT-based plotting
│   ├── roc/                  # (NEW) ROC curve analysis
│   ├── sampletracer/         # (NEW) Multi-file orchestrator
│   ├── selection/
│   ├── structs/              # (NEW) Fundamental data structures
│   ├── tools/
│   └── typecasting/          # (NEW) Type conversion utilities
├── pyc/                      # PyC CUDA modules
│   ├── pyc_overview.dox
│   ├── cutils/
│   ├── graph/
│   ├── interface/            # (NEW) PyTorch C++ bindings
│   ├── nusol/
│   ├── operators/
│   ├── physics/
│   └── transform/
└── templates/                # Template base classes
    ├── event_template.dox
    └── particle_template.dox
```

## Documentation Coverage

### ✅ Documented Modules (29/34)

The core infrastructure of the framework is fully documented.

**Core Modules (19):**
- analysis, container, dataloader, graph, io, lossfx, meta, metric, metrics
- model, notification, optimizer, plotting, roc, sampletracer, selection
- structs, tools, typecasting

**PyC CUDA Modules (7):**
- cutils, graph, interface, nusol, operators, physics, transform

**Template Classes (2):**
- event_template, particle_template

### 🟡 Placeholder Documentation (5/34)

Placeholder files have been created for the following implementation-specific modules. These files need to be populated with detailed documentation.

- **events**: Contains specific event implementations.
- **graphs**: Contains specific graph-building logic.
- **selections**: Contains numerous event selection algorithms.
- **models**: Contains GNN model implementations.
- **metrics**: Contains specific metric calculators.


### 📋 Optional/Specialized

**nusol/ subdirectories (ellipse, conuix, tmp):**
- Highly specialized mathematical solvers (elliptic integrals, conic sections)
- Users should reference scientific papers for theoretical background
- Not required for general framework usage

## Recent Additions (November 2025)

### New Module Documentation

1. **container.dox** (~620 lines)
   - File-level data container with hash-based entry management
   - Template instance coordination
   - Dataloader population

2. **sampletracer.dox** (~700 lines)
   - Multi-file orchestrator
   - Parallel graph compilation
   - Cross-file selection application

3. **plotting.dox** (~580 lines)
   - Matplotlib-inspired ROOT visualization
   - Variable binning and error bars
   - Cross-section weighting

4. **roc.dox** (~650 lines)
   - ROC curve analysis
   - Binary and multi-class support
   - AUC calculation via trapezoidal integration

5. **metrics.dox** (~1000 lines)
   - Training analytics with ROOT histograms
   - TGraph-based progress plots
   - K-fold metric aggregation

6. **structs.dox** (~1800 lines)
   - Covers all 12 fundamental header files
   - Type system (bsc_t, data_enum)
   - ROOT I/O structures
   - Training configuration structures

7. **typecasting.dox** (~1100 lines)
   - Tensor ↔ vector conversions
   - ROOT TTree integration (variable_t)
   - Ragged array padding

8. **pyc/interface/interface.dox** (~900 lines)
   - PyTorch C++ bindings for CUDA kernels
   - neutrino class documentation
   - All PyC namespace documentation

## Building Documentation

### Prerequisites

```bash
# Install Doxygen
sudo apt-get install doxygen graphviz

# Or via Homebrew (macOS)
brew install doxygen graphviz
```

### Generate HTML Documentation

```bash
cd /workspaces/AnalysisG
doxygen Doxyfile
```

### View Documentation

```bash
# Open in browser
firefox doxygen-docs/html/index.html

# Or
xdg-open doxygen-docs/html/index.html
```

## Documentation Standards

### File Format

All documentation files use the `.dox` extension and follow Doxygen syntax:

```cpp
/**
 * @file module_name.dox
 * @brief Brief description
 * @defgroup module_name Module Name
 * @details
 * Detailed description of the module.
 *
 * @page module_name_page Module Name Documentation
 *
 * @section intro Introduction
 * ...
 */
```

### Section Structure

Each module documentation includes:

1. **Introduction**: Purpose and key features
2. **Class/Structure Definitions**: API documentation
3. **Workflow**: Usage patterns and integration
4. **Implementation Details**: Internal mechanics
5. **Usage Examples**: Code snippets with explanations
6. **Related Modules**: Cross-references
7. **Summary**: Quick reference

### Cross-Referencing

- Use `@ref module_name` for module links
- Use `@ref ClassName` for class references
- Use `@see` for related documentation

## Key Documentation Pages

- **Main Index**: `modules_index.dox` - Complete module catalog
- **Overview**: `overview.dox` - Framework pipeline and workflow
- **Interactions**: `module_interactions.dox` - Dependency graphs

## Doxygen Configuration

The `Doxyfile` in the repository root controls:

- **Input Sources**: `src/AnalysisG` + `docs/doxygen`
- **File Patterns**: `*.h *.cxx *.cu *.cuh *.pyx *.pxd *.dox`
- **Output**: `doxygen-docs/html/`
- **Features**: Source browser, call graphs, treeview navigation
- **Cython Support**: `EXTENSION_MAPPING = pyx=C++ pxd=C++` (treats Cython as C++)

### Automatic Processing

**C++ Core Documentation** (.h, .cxx, .cu, .cuh):
- Extracted directly from C++ source files
- Classes, functions, templates documented via Doxygen comments

**Cython Python Bindings** (.pyx, .pxd):
- **53 .pyx files**: Python wrapper implementations (auto-processed)
- **100 .pxd files**: Cython header declarations (auto-processed)
- Mapped to C++ via `EXTENSION_MAPPING` in Doxyfile
- No separate .dox files needed - documentation extracted from source

**Module Documentation** (.dox):
- High-level module overviews and usage examples
- Workflow documentation and integration guides
- 28 comprehensive module documentation files

## Contributing

When adding new modules:

1. Create `.dox` file in appropriate directory (modules/, pyc/, templates/)
2. Follow existing documentation structure
3. Include comprehensive examples
4. Add cross-references to related modules
5. Update `modules_index.dox` with new module entry

## Maintenance

- **Regular Updates**: Keep documentation synchronized with code changes
- **Validation**: Run `doxygen Doxyfile` to check for warnings
- **Coverage**: Aim for 100% documentation of public APIs
- **Examples**: Ensure all examples compile and run correctly

## Contact

For documentation questions or contributions:
- **Repository**: https://github.com/woywoy123/AnalysisG
- **Pull Requests**: Create PR with documentation updates
- **Issues**: Report documentation gaps or errors

---

**Last Updated**: November 17, 2025
**Documentation Coverage**: 29/34 modules (~85%)
**Total Documentation**: ~8500 lines across all modules

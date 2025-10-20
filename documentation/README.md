# AnalysisG Documentation

This directory contains comprehensive Doxygen documentation for the AnalysisG framework.

## Structure

```
documentation/
├── .readthedocs.yaml          # Read the Docs configuration
├── Doxyfile                    # Doxygen configuration file
├── conf.py                     # Sphinx configuration
├── index.rst                   # Main documentation index
├── requirements.txt            # Python dependencies for documentation
├── README.md                   # This file
│
├── *.dox files                 # Module documentation files
│   ├── mainpage.dox           # Main documentation page
│   ├── analysis.dox           # Analysis module documentation
│   ├── particle.dox           # Particle module documentation
│   ├── event.dox              # Event module documentation
│   ├── graph.dox              # Graph module documentation
│   ├── dataloader.dox         # DataLoader module documentation
│   ├── model_selection_io.dox # Model, Selection, I/O modules
│   ├── lossfx_metric_metrics.dox # Loss, Metric modules
│   ├── container_meta_notification_tools.dox # Support modules
│   └── remaining_modules.dox  # Other modules
│
└── output/                     # Generated documentation (created by Doxygen)
    ├── html/                   # HTML output
    ├── xml/                    # XML output (for Breathe/Sphinx)
    └── analysisg.tag          # Tag file

```

## Documentation Modules

The documentation is organized by module, covering all components of the AnalysisG framework:

### Core Modules
- **Analysis**: Main orchestrator for physics analysis workflow
- **Particle**: Particle representation with 4-vector support
- **Event**: Event data structures and processing
- **Graph**: Graph representations for GNN applications

### Data Management
- **DataLoader**: Data batching, k-fold CV, caching
- **Container**: Data organization and management
- **I/O**: ROOT and HDF5 file operations
- **Meta**: Metadata handling

### Machine Learning
- **Model**: ML model templates and PyTorch integration
- **Loss Functions**: Comprehensive loss functions and optimizers
- **Metric**: Performance metrics templates
- **Metrics**: Built-in physics metrics

### Analysis Tools
- **Selection**: Event and object selection framework
- **Plotting**: Visualization utilities
- **SampleTracer**: Sample tracking and processing

### Infrastructure
- **Notification**: Logging and messaging system
- **Tools**: Utility functions
- **Structs**: Core data structures and types
- **TypeCasting**: Type conversion utilities
- **Variable**: Variable management
- **XML**: XML parsing and configuration

## Building Documentation Locally

### Prerequisites

```bash
# Install Doxygen
sudo apt-get install doxygen graphviz

# Install Python dependencies
pip install -r requirements.txt
```

### Generate Doxygen Documentation

```bash
# From the documentation directory
doxygen Doxyfile
```

This will generate:
- HTML documentation in `output/html/`
- XML files in `output/xml/` (for Sphinx integration)

View the HTML documentation by opening `output/html/index.html` in a browser.

### Generate Sphinx Documentation (with Doxygen integration)

```bash
# From the documentation directory
sphinx-build -b html . _build
```

This will:
1. Run Doxygen automatically via conf.py
2. Use Breathe to integrate Doxygen XML into Sphinx
3. Generate complete documentation in `_build/`

View the Sphinx documentation by opening `_build/index.html` in a browser.

## Read the Docs Integration

This documentation is configured to be hosted on Read the Docs. The configuration file `.readthedocs.yaml` specifies:

- Build environment (Ubuntu 22.04, Python 3.11)
- Required packages (doxygen, graphviz, cmake, g++)
- Python requirements
- Sphinx configuration
- Output formats (HTML, PDF, EPUB)

### Hosting on Read the Docs

1. Import the project on Read the Docs
2. Point to this repository
3. The build will automatically:
   - Install dependencies
   - Run Doxygen
   - Build Sphinx documentation
   - Generate HTML, PDF, and EPUB outputs

## Documentation Features

- **Comprehensive Coverage**: All modules, classes, functions documented
- **Cross-References**: Links between related components
- **Code Examples**: Usage examples for each module
- **Graphs**: Class diagrams, collaboration graphs, call graphs
- **Search**: Full-text search functionality
- **Multi-Format**: HTML, PDF, EPUB outputs
- **API Reference**: Complete API documentation
- **Module Organization**: Clear grouping by functionality

## Contributing to Documentation

When adding or modifying code:

1. **Add Doxygen comments** to headers (.h files):
   ```cpp
   /**
    * @brief Brief description
    * @details Detailed description
    * @param param_name Parameter description
    * @return Return value description
    */
   ```

2. **Update .dox files** if adding new modules or major features

3. **Test documentation** locally before committing:
   ```bash
   doxygen Doxyfile
   # Check output/html/index.html
   ```

4. **Follow existing style**: See current .dox files for examples

## Documentation Style Guide

- Use `@brief` for short descriptions
- Use `@details` for longer explanations
- Use `@param` for parameters
- Use `@return` for return values
- Use `@see` for cross-references
- Use `@note` for important notes
- Use `@warning` for warnings
- Group related items with `@defgroup` and `@ingroup`

## Files Created

This documentation structure includes the following files:

1. `mainpage.dox` - Main documentation page with overview
2. `analysis.dox` - Analysis module documentation
3. `particle.dox` - Particle module documentation
4. `event.dox` - Event module documentation
5. `graph.dox` - Graph module documentation
6. `dataloader.dox` - DataLoader module documentation
7. `model_selection_io.dox` - Model, Selection, and I/O modules
8. `lossfx_metric_metrics.dox` - Loss functions and metrics modules
9. `container_meta_notification_tools.dox` - Support modules
10. `remaining_modules.dox` - Other framework modules
11. `Doxyfile` - Doxygen configuration
12. `.readthedocs.yaml` - Read the Docs configuration
13. `requirements.txt` - Python dependencies
14. `conf.py` - Sphinx configuration
15. `index.rst` - Sphinx main index
16. `README.md` - This file

## License

See the main repository LICENSE file.

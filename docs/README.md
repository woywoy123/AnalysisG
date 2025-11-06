# AnalysisG Documentation

This directory contains the comprehensive documentation for the AnalysisG framework.

## Overview

The documentation is built using Sphinx with the following components:

- **Sphinx**: Main documentation system
- **Breathe**: C++ documentation integration via Doxygen (optional, for local builds with Doxygen)
- **Read the Docs Theme**: Professional documentation theme

**Note**: The documentation builds successfully without Doxygen/Breathe. C++ API documentation via Breathe is only available when Doxygen is run locally.

## Structure

```
docs/
├── source/                    # Documentation source files (RST)
│   ├── index.rst             # Main documentation index
│   ├── introduction.rst      # Framework introduction
│   ├── installation.rst      # Installation guide
│   ├── quickstart.rst        # Quick start guide
│   ├── interfaces/           # Simple user-facing interfaces
│   ├── technical/            # Complex technical components
│   ├── core/                 # Core package documentation
│   ├── events/               # Events package documentation
│   ├── graphs/               # Graphs package documentation
│   ├── metrics/              # Metrics package documentation
│   ├── models/               # Models package documentation
│   ├── modules/              # Modules package documentation (C++)
│   ├── pyc/                  # PyC package documentation (C++/CUDA)
│   ├── templates/            # Templates documentation
│   ├── _static/              # Static files (images, CSS, etc.)
│   └── _templates/           # Sphinx templates
├── deprecated/               # Old documentation (preserved for reference)
├── Doxyfile                  # Doxygen configuration for C++ API
├── conf.py                   # Sphinx configuration
├── requirements.txt          # Documentation dependencies
└── .readthedocs.yaml         # Read the Docs configuration

## Building Documentation

### Prerequisites

```bash
pip install -r requirements.txt
```

Optional for C++ API documentation:
- Doxygen (for C++ API docs)

### Build HTML Documentation

```bash
cd docs
make html
```

The generated documentation will be in `docs/_build/html/`.

### Build C++ API Documentation

```bash
cd docs
doxygen Doxyfile
```

This generates XML output in `docs/xml/` which is automatically integrated by Breathe.

### Build PDF Documentation

```bash
cd docs
make latexpdf
```

### Clean Build

```bash
cd docs
make clean
```

## Documentation Categories

### Simple Interfaces (User-Facing)

Documentation for high-level APIs that users inherit and customize:

- **EventTemplate**: Event-level data structures
- **ParticleTemplate**: Particle-level data structures
- **GraphTemplate**: Graph representations
- **MetricTemplate**: Evaluation metrics
- **ModelTemplate**: Machine learning models
- **SelectionTemplate**: Selection criteria

These are designed to be extended by users for their specific analyses.

### Complex Technical Components

Documentation for low-level C++ implementations:

- **Modules Package**: Core C++ implementations
  - Analysis framework
  - Data containers
  - Event processing
  - Graph algorithms
  - I/O operations
  - Optimization algorithms
  - And 17 other modules

- **PyC Package**: Python-C++/CUDA interface
  - CUDA utilities
  - GPU-accelerated graph operations
  - Neutrino reconstruction
  - Physics calculations
  - Coordinate transformations

These are typically used internally and provide high-performance backends.

## Writing Documentation

### Adding New Documentation

1. Create a new `.rst` file in the appropriate directory under `source/`
2. Add the file to the relevant `index.rst` or overview file
3. Follow the existing style and structure
4. Include code examples where appropriate

### RST Syntax Examples

#### Sections

```rst
Main Title
==========

Section
-------

Subsection
~~~~~~~~~~

Subsubsection
^^^^^^^^^^^^^
```

#### Code Blocks

```rst
.. code-block:: python

   from AnalysisG.core import EventTemplate
   
   class MyEvent(EventTemplate):
       pass
```

#### Cross-References

```rst
See :doc:`other_file` for more information.
See :class:`EventTemplate` for the class documentation.
```

## Continuous Integration

Documentation is automatically built and deployed by Read the Docs when changes are pushed to the repository.

- **URL**: https://analysisg.readthedocs.io/
- **Build Status**: Visible in Read the Docs dashboard

## Contributing

When contributing documentation:

1. Keep language clear and concise
2. Include code examples for all APIs
3. Document all parameters and return values
4. Add cross-references to related documentation
5. Test that documentation builds without errors
6. Ensure C++ documentation is up-to-date

## License

The documentation is licensed under the same terms as the AnalysisG project (see LICENSE file in the repository root).

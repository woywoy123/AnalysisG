# AnalysisG Documentation

This directory contains the documentation for the AnalysisG framework.

## Structure

- `source/` - Sphinx source files
  - `api/` - Auto-generated API documentation
  - `doxygen/` - Generated .dox files documenting source code
  - Other directories contain manually written documentation
- `deprecated/` - Previous documentation (moved from source/)
- `old/` - Legacy documentation
- `Doxyfile` - Doxygen configuration
- `requirements.txt` - Python dependencies for building docs

## Building Documentation

### Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
sudo apt-get install doxygen graphviz  # On Ubuntu/Debian
```

### Build Process

The documentation uses both Sphinx and Doxygen:

1. **Doxygen** generates detailed API documentation from C++ source files
2. **Sphinx** builds the main documentation and integrates Doxygen output via Breathe

To build:

```bash
cd docs
make html
```

The HTML documentation will be in `build/html/index.html`.

To build only Doxygen documentation:

```bash
cd docs
doxygen Doxyfile
```

The Doxygen HTML will be in `build/doxygen/html/index.html`.

## Auto-Generated Documentation

The `.dox` files in `source/doxygen/` are automatically generated from the source code using a script. They provide:

- Class documentation
- Function signatures
- File dependencies
- Module overviews

These files are merged for related source files (e.g., `.h` and `.cxx` files with the same base name).

## Documentation Workflow

1. **Source Code** - C++, Python files in `src/AnalysisG/`
2. **Generation Script** - Scans source and creates `.dox` files
3. **Doxygen** - Processes source + `.dox` files → XML
4. **Breathe** - Sphinx extension reads Doxygen XML
5. **Sphinx** - Builds final HTML documentation

## Read the Docs Integration

The documentation is automatically built and published on Read the Docs:
- Configuration: `.readthedocs.yaml` in the root
- Build command: `make html` 
- Doxygen runs automatically during Sphinx build (via `conf.py` setup hook)

## Notes

- The `conf.py` setup hook runs Doxygen before Sphinx build
- Build artifacts in `build/` are gitignored
- Template files with `<name>` placeholders may cause XML parsing issues
- API documentation pages avoid directly embedding problematic classes

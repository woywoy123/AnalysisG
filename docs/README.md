# AnalysisG Documentation

This directory contains the documentation infrastructure for AnalysisG, using Doxygen for C++ API documentation and Sphinx with Breathe for the overall documentation website.

## Documentation Structure

```
docs/
├── doxygen/           # Doxygen documentation files (.dox)
│   ├── mainpage.dox   # Main landing page
│   ├── core.dox       # Core module documentation
│   ├── events.dox     # Events module documentation
│   ├── graphs.dox     # Graphs module documentation
│   ├── metrics.dox    # Metrics module documentation
│   ├── models.dox     # Models module documentation
│   ├── modules.dox    # Infrastructure modules documentation
│   ├── pyc.dox        # Python-C++ interface documentation
│   └── templates.dox  # Templates module documentation
│
├── source/            # Sphinx documentation source files
│   ├── conf.py        # Sphinx configuration
│   ├── index.rst      # Main documentation index
│   ├── introduction.rst
│   ├── examples.rst
│   ├── contributing.rst
│   ├── api/           # API reference documentation
│   │   └── *.rst      # Per-module API documentation
│   └── modules/       # High-level module documentation
│       └── *.rst      # Per-module guides
│
├── build/             # Sphinx build output (not in git)
│   └── html/          # HTML documentation
│
├── requirements.txt   # Python dependencies for building docs
├── Makefile          # Build automation for Unix
└── make.bat          # Build automation for Windows
```

## Building Documentation

### Prerequisites

1. **Doxygen** (for C++ API documentation)
   ```bash
   sudo apt-get install doxygen graphviz
   ```

2. **Python packages** (for Sphinx)
   ```bash
   pip install -r requirements.txt
   ```

### Build Steps

#### 1. Generate Doxygen Documentation

From the repository root:

```bash
doxygen Doxyfile
```

This creates:
- `doxygen-docs/html/` - HTML documentation (browseable)
- `doxygen-docs/xml/` - XML output (used by Breathe)

#### 2. Build Sphinx Documentation

From the `docs/` directory:

```bash
cd docs
make html
```

Or on Windows:

```cmd
cd docs
make.bat html
```

The generated HTML documentation will be in `docs/build/html/`.

#### 3. View Documentation

Open `docs/build/html/index.html` in your web browser.

Or for Doxygen-only documentation:

Open `doxygen-docs/html/index.html` in your web browser.

## Documentation Components

### Doxygen (.dox files)

The `.dox` files in `docs/doxygen/` contain:
- Module group definitions
- High-level documentation
- Usage examples
- Design philosophy

These files are processed by Doxygen along with the source code comments.

### Sphinx (RST files)

The `.rst` files in `docs/source/` provide:
- User-facing documentation
- Tutorials and examples
- Getting started guides
- Contribution guidelines

### Integration

Sphinx uses the **Breathe** extension to integrate Doxygen's XML output, providing:
- C++ API documentation in Sphinx pages
- Cross-references between user docs and API docs
- Unified search across all documentation

## Read the Docs Integration

The documentation is configured for automatic building on Read the Docs via `.readthedocs.yaml` in the repository root.

When changes are pushed to the repository:
1. Read the Docs runs `doxygen Doxyfile`
2. Then runs `sphinx-build` to generate final HTML
3. Publishes the result at the project's RTD URL

## Maintenance

### Adding New Modules

When adding a new module to AnalysisG:

1. Create a `.dox` file in `docs/doxygen/` with module documentation
2. Add corresponding `.rst` files in `docs/source/api/` and `docs/source/modules/`
3. Update the table of contents in relevant index files
4. Rebuild the documentation

### Updating API Documentation

API documentation is extracted directly from C++ source code comments. Use Doxygen comment style:

```cpp
/**
 * @brief Brief description of the class
 * 
 * Detailed description of what this class does
 * and how to use it.
 */
class MyClass {
public:
    /**
     * @brief Brief method description
     * @param param1 Description of first parameter
     * @param param2 Description of second parameter
     * @return Description of return value
     */
    int myMethod(int param1, float param2);
};
```

## Troubleshooting

### Doxygen Warnings

If Doxygen produces warnings about undocumented code:
- This is normal for existing code
- Document public APIs as you work with them
- Warnings are disabled for undocumented entities in the Doxyfile

### Sphinx Build Errors

Common issues:

1. **Missing dependencies**: Install requirements.txt
2. **Breathe errors**: Ensure Doxygen XML is generated first
3. **RST syntax errors**: Check the error message for line numbers

### Read the Docs Build Failures

Check:
1. `.readthedocs.yaml` configuration is correct
2. All dependencies are listed in `requirements.txt`
3. Build logs on Read the Docs dashboard for specific errors

## Contributing to Documentation

See [CONTRIBUTING.md](../CONTRIBUTING.md) for general contribution guidelines.

For documentation specifically:
- Use clear, concise language
- Include code examples where helpful
- Keep API docs close to the code (in source comments)
- Keep user guides in RST files
- Test your changes by building locally

## Additional Resources

- [Doxygen Manual](https://www.doxygen.nl/manual/)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Breathe Documentation](https://breathe.readthedocs.io/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)

# Developer Guide

This section provides information for developers who want to contribute to AnalysisG or extend its functionality.

## Development Setup

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/AnalysisG.git
   cd AnalysisG
   ```

2. Set up a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

## Code Structure

The AnalysisG codebase is organized as follows:

- `analysisg/` - Main package source code
- `tests/` - Unit and integration tests
- `docs/` - Documentation
- `studies/` - Example studies and implementations
- `scripts/` - Utility scripts

## Coding Standards

- Follow PEP 8 style guidelines
- Use type annotations
- Document all public functions and classes using docstrings
- Write unit tests for new functionality

## Test Suite

Run the test suite using pytest:

```bash
pytest tests/
```

For test coverage:

```bash
pytest --cov=analysisg tests/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Documentation

Documentation is written in Markdown and built using MkDocs. To build the docs:

```bash
mkdocs build
```

To serve the documentation locally:

```bash
mkdocs serve
```

## Release Process

Releasing a new version:

1. Update version in `setup.py`
2. Update changelog
3. Create a new git tag
4. Build and publish to PyPI
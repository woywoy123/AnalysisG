# AnalysisG Code Examples

This directory contains example code snippets extracted from the `studies` directory of the AnalysisG framework. These examples demonstrate various usage patterns and techniques for using the framework's components.

## Directory Structure

Examples are organized by their source module in the studies directory:

- `analysis/` - Examples from analysis studies
- `benchmarks/` - Examples of benchmarking code
- `mc16_matching/` - Examples of MC16 matching techniques 
- `mc20_experimental/` - Examples of MC20 experimental code
- `metrics/` - Examples of metric calculations
- `neutrino/` - Examples of neutrino reconstruction
- `performance/` - Examples of performance measurement code
- `topreconstruction/` - Examples of top quark reconstruction algorithms

## How to Use These Examples

Each example file includes:
- Source file path information
- Import statements from the original file
- Class definitions
- Useful functions 
- Entry points or main execution blocks

You can use these examples as a reference when developing your own analysis with AnalysisG.

## Running the Examples

Most examples require the AnalysisG framework to be properly installed and configured. Some examples might also require specific data files or additional dependencies.

To use these examples:

1. Make sure you have AnalysisG installed
2. Copy the relevant code into your project
3. Adjust paths and configurations as needed for your environment
4. Run the script

## Generating Examples

These examples are automatically generated using the script at `/workspaces/AnalysisG/scripts/generate_examples.py`. To regenerate or update the examples, run:

```bash
python /workspaces/AnalysisG/scripts/generate_examples.py
```
# First Steps with AnalysisG

This tutorial will guide you through understanding how to use AnalysisG based on actual working examples from the framework.

!!! note "Prerequisites"
    This guide assumes you have already installed AnalysisG following the [installation instructions](installation.md). The examples shown here are based on actual working code from the framework's `studies/` and tutorial directories.

## Overview

A typical AnalysisG workflow consists of:

1. **Import Framework** - Import the Analysis class and required modules
2. **Define/Use Event Templates** - Use pre-defined or create custom event structures
3. **Define/Use Graph Templates** (optional) - For machine learning applications
4. **Add Selections** (optional) - Filter events based on physics criteria
5. **Configure Analysis** - Set up samples, events, graphs, and selections
6. **Run Analysis** - Process the data with `Start()`
7. **Visualize Results** - Use plotting utilities

## Basic Example: Setting Up an Analysis

### Step 1: Import Required Modules

Based on actual usage from the framework:

```python
from AnalysisG import Analysis
from AnalysisG.events.gnn import EventGNN  # Pre-built event template
from AnalysisG.core.plotting import TH1F, TH2F
```

### Step 2: Create and Configure Analysis

```python
# Create analysis object
ana = Analysis()

# Create event template instance
ev = EventGNN()

# Add event template with a label
ana.AddEvent(ev, "gnn")

# Add input ROOT files with matching label
ana.AddSamples("/path/to/data.root", "gnn")

# Set number of processing threads
ana.Threads = 4

# Set output path for results
ana.OutputPath = "./output/"
```

### Step 3: Add Selections (Optional)

```python
from AnalysisG.selections.performance.topefficiency import TopEfficiency

# Create and add selection
sel = TopEfficiency()
ana.AddSelection(sel)
```

### Step 4: Run the Analysis

```python
# Start processing
ana.Start()

# Access results from selection object
# Results are stored in the selection instance
```

## Working with Graphs for Machine Learning

AnalysisG can create graph representations of events for GNN training:

```python
from AnalysisG import Analysis
from AnalysisG.events.bsm_4tops import BSM4Tops
from AnalysisG.graphs.bsm_4tops import GraphJets

# Setup analysis
ana = Analysis()
ev = BSM4Tops()
gr = GraphJets()

# Add components
ana.AddEvent(ev, "bsm")
ana.AddGraph(gr, "jets")
ana.AddSamples("/path/to/data.root", "bsm")

# Configure and run
ana.Threads = 4
ana.Start()
```

## Creating Plots

Use the plotting utilities after processing:

```python
from AnalysisG.core.plotting import TH1F

# Create histogram with data
hist = TH1F(
    Title="Distribution Title",
    xData=[1.0, 2.0, 3.0, 4.0, 5.0],  # Your data
    xTitle="Variable [units]",
    yTitle="Events",
    Color="blue",
    Density=True,
    ErrorBars=True
)

# Configure output
hist.Filename = "my_histogram"
hist.OutputDirectory = "./plots/"
hist.DPI = 300

# Set style (optional)
hist.Style = "ATLAS"  # Use ATLAS collaboration style

# Save the figure
hist.SaveFigure()
```

## Complete Working Example

Based on the actual tutorial in the framework:

```python
import os
from AnalysisG import Analysis
from AnalysisG.events.gnn import EventGNN
from AnalysisG.selections.performance.topefficiency import TopEfficiency
from AnalysisG.core.plotting import TH1F

# Configuration
DATA_PATH = "/path/to/your/data.root"
OUTPUT_DIR = "./output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create components
sel = TopEfficiency()
ana = Analysis()
ev = EventGNN()

# Configure analysis
ana.Threads = 4
ana.OutputPath = OUTPUT_DIR
ana.AddEvent(ev, "gnn")
ana.AddSelection(sel)
ana.AddSamples(DATA_PATH, "gnn")

# Run analysis
print("Starting analysis...")
ana.Start()
print("Analysis finished!")

# Save results
output_file = os.path.join(OUTPUT_DIR, "results.pkl")
sel.dump(output_file)
print(f"Results saved to {output_file}")

# Create a plot from results
# (Note: Actual plotting depends on what data your selection stores)
hist = TH1F(
    Title="Analysis Results",
    xTitle="Observable [units]",
    yTitle="Events",
    Density=True
)
hist.Filename = "results"
hist.OutputDirectory = OUTPUT_DIR
# hist.SaveFigure()  # Uncomment when you have data to plot
```

## Important Notes

### About These Examples

The examples in this guide are based on real working code from the AnalysisG framework but are **simplified for illustration**. In practice:

1. **Event Templates**: The framework provides pre-built event templates (like `EventGNN`, `BSM4Tops`) that handle complex ROOT file structures. Creating custom event templates requires understanding the C++/Cython interface.

2. **Data Requirements**: These examples assume you have ROOT files containing appropriate physics data. The framework is designed for LHC (Large Hadron Collider) data analysis.

3. **Selection Objects**: The `TopEfficiency` and other selection classes shown are complex analysis tools with their own internal logic for physics reconstruction and filtering.

4. **Results Access**: How you access results depends on what your selection class stores. Check the specific selection class documentation for available attributes.

### Testing Your Installation

To verify your installation works, you can run the existing tests:

```bash
# Navigate to test directory
cd pyc/test/

# Run a simple test (if dependencies are met)
python3 test_operators.py
```

## Next Steps

Now that you understand the basics, explore:

- [User Guide](../user_guide/index.md) - Detailed documentation of all features
- [API Reference](../api/index.md) - Complete API documentation
- [Examples](../examples/README.md) - Real analysis examples from `studies/` directory
- [Complete Tutorial](../_docs/templates/tutorial/complete_analysis.py) - Full working example

## Getting Help

- Check the existing examples in the `studies/` directory of the repository
- Review the [troubleshooting guide](../_docs/misc/troubleshooting.rst)
- Review the complete tutorial at `docs/_docs/templates/tutorial/complete_analysis.py`
- Open an issue on [GitHub](https://github.com/woywoy123/AnalysisG/issues)

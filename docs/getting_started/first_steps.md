# First Steps with AnalysisG

This tutorial will guide you through your first analysis using AnalysisG, from loading data to producing results.

## Overview

A typical AnalysisG workflow consists of:

1. **Define Event Structure** - Describe your input data format
2. **Define Particle Objects** - Specify physics objects (jets, leptons, etc.)
3. **Apply Selections** - Filter events based on physics criteria
4. **Create Graphs** (optional) - Build graph representations for ML
5. **Run Analysis** - Process the data
6. **Visualize Results** - Create plots and histograms

## Basic Example: Event Processing

### Step 1: Import AnalysisG

```python
import AnalysisG as ag
```

### Step 2: Define an Event Template

Event templates describe the structure of your input ROOT files:

```python
class MyEvent(ag.EventTemplate):
    """Custom event class for my analysis."""
    
    def __init__(self):
        super().__init__()
        # Define branches to read from ROOT file
        self.add_branch("EventNumber", int)
        self.add_branch("Jets_pt", "vector<float>")
        self.add_branch("Jets_eta", "vector<float>")
        self.add_branch("Jets_phi", "vector<float>")
        self.add_branch("Jets_m", "vector<float>")
```

### Step 3: Create an Analysis

```python
# Create analysis object
ana = ag.Analysis()

# Add input files
ana.AddSamples([
    "/path/to/data/sample1.root",
    "/path/to/data/sample2.root"
])

# Specify the tree name in ROOT file
ana.TreeName = "CollectionTree"

# Set the event template
ana.EventCache = MyEvent
```

### Step 4: Process Events

```python
# Process events
ana.Run()

# Access processed events
events = ana.EventCache

# Print number of events
print(f"Processed {len(events)} events")
```

## Working with Particle Objects

AnalysisG provides built-in physics object classes:

```python
# Access jets from an event
for event in events:
    for jet in event.jets:
        print(f"Jet pT: {jet.pt:.2f} GeV")
        print(f"Jet eta: {jet.eta:.2f}")
        print(f"Jet phi: {jet.phi:.2f}")
```

## Applying Selections

Create selection classes to filter events:

```python
class JetSelection(ag.SelectionTemplate):
    """Select events with at least 4 jets."""
    
    def selection(self, event):
        # Count jets with pT > 25 GeV and |eta| < 2.5
        n_jets = sum(1 for jet in event.jets 
                     if jet.pt > 25 and abs(jet.eta) < 2.5)
        return n_jets >= 4

# Apply selection
ana.AddSelection(JetSelection)
ana.Run()

print(f"Events passing selection: {ana.EventsPassed}")
```

## Creating Histograms

Use the plotting utilities to create histograms:

```python
from AnalysisG.core.plotting import TH1F

# Create a histogram
hist = TH1F()
hist.Title = "Jet pT Distribution"
hist.xTitle = "Jet pT [GeV]"
hist.yTitle = "Events"
hist.Filename = "jet_pt"
hist.OutputDirectory = "./plots"

# Fill histogram
for event in events:
    for jet in event.jets:
        hist.Fill(jet.pt)

# Save the plot
hist.SaveFigure()
```

## Complete Example

Here's a complete example putting it all together:

```python
import AnalysisG as ag
from AnalysisG.core.plotting import TH1F

# Define event structure
class MyEvent(ag.EventTemplate):
    def __init__(self):
        super().__init__()
        self.add_branch("Jets_pt", "vector<float>")
        self.add_branch("Jets_eta", "vector<float>")
        self.add_branch("Jets_phi", "vector<float>")

# Define selection
class BasicSelection(ag.SelectionTemplate):
    def selection(self, event):
        return len(event.jets) >= 4

# Create analysis
ana = ag.Analysis()
ana.AddSamples(["/path/to/data.root"])
ana.TreeName = "CollectionTree"
ana.EventCache = MyEvent
ana.AddSelection(BasicSelection)

# Run analysis
ana.Run()

# Create histogram
hist = TH1F()
hist.Title = "Leading Jet pT"
hist.xTitle = "pT [GeV]"
hist.yTitle = "Events"

# Fill histogram with leading jet pT
for event in ana.EventCache:
    if event.jets:
        hist.Fill(event.jets[0].pt)

# Save plot
hist.Filename = "leading_jet_pt"
hist.OutputDirectory = "./output"
hist.SaveFigure()

print(f"Analysis complete! Processed {len(ana.EventCache)} events")
```

## Next Steps

Now that you understand the basics, explore:

- [User Guide](../user_guide/index.md) - Detailed documentation of all features
- [API Reference](../api/index.md) - Complete API documentation
- [Examples](../examples/README.md) - More complex examples and tutorials

## Getting Help

- Check the [troubleshooting guide](../misc/troubleshooting.rst)
- Review the [API documentation](../api/index.md)
- Open an issue on [GitHub](https://github.com/woywoy123/AnalysisG/issues)

# Example using AnalysisG.core.plotting.TH1F
# ============================================================

import numpy as np
from AnalysisG.core.plotting import TH1F

# --- Create Sample Data ---
# Generate some random data points (e.g., representing particle energies)
np.random.seed(42) # for reproducibility
data_values = np.random.normal(loc=100, scale=20, size=1000)

# Generate corresponding weights (e.g., event weights)
weights = np.random.uniform(0.5, 1.5, size=1000)

# --- Create and Configure Histogram ---
# Instantiate the histogram object
hist = TH1F()

# Set histogram properties
hist.Title = "Simple Example Histogram"
hist.xData = data_values
hist.Weights = weights
hist.xBins = 20       # Number of bins
hist.xMin = 0         # Minimum x-axis value
hist.xMax = 200       # Maximum x-axis value
hist.xTitle = "Energy (GeV)"
hist.yTitle = "Weighted Events / Bin"
hist.Color = "blue"
hist.Filename = "simple_histogram" # Output filename (without extension)
hist.ShowCount = True # Show number of entries on the plot
hist.ErrorBars = True # Show statistical error bars

# --- Save the Histogram Figure ---
# This will generate a plot file (e.g., simple_histogram.png)
hist.SaveFigure()

print("Simple histogram 'simple_histogram.png' created.")

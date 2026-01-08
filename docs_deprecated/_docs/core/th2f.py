# Import necessary classes from the AnalysisG plotting library
from AnalysisG.core.plotting import TH1F, TH2F
import random # For generating example data

# --- Data Preparation (Example based on the provided code's logic) ---
# In a real scenario, this data would come from your analysis objects (like ttbar, tttt)
# We'll simulate some data for demonstration purposes.

# Simulate PageRank scores and masses for two processes (e.g., ttbar and tttt)
ttbar_scores = [random.uniform(0, 1) for _ in range(500)]
ttbar_masses = [random.gauss(172.5, 20) for _ in range(500)]

tttt_scores = [random.uniform(0, 1) for _ in range(200)]
tttt_masses = [random.gauss(172.5, 30) for _ in range(200)] # Wider distribution for variety

truth_masses = [random.gauss(172.5, 5) for _ in range(700)] # Simulated truth distribution

# Define output directory
output_dir = "./output_tutorial"

# --- Tutorial: Creating Plots ---

# 1. Simple 1D Histogram (TH1F) - PageRank Scores (Normalized)

# Create histogram objects for each process
hist_ttbar_score = TH1F()
hist_ttbar_score.xData = [s for s in ttbar_scores if s < 0.99] # Apply a cut like in the example
hist_ttbar_score.Color = "red"
hist_ttbar_score.Title = r'$\bar{t}t$ Scores' # Use raw strings for LaTeX
hist_ttbar_score.Density = True      # Normalize the histogram area to 1
hist_ttbar_score.ErrorBars = True    # Show statistical error bars

hist_tttt_score = TH1F()
hist_tttt_score.xData = [s for s in tttt_scores if s < 0.99]
hist_tttt_score.Color = "blue"
hist_tttt_score.Title = r'$\bar{t}t\bar{t}t$ Scores'
hist_tttt_score.Density = True
hist_tttt_score.ErrorBars = True

# Create the main plot object to combine the histograms
plot_pagerank = TH1F()
plot_pagerank.Histograms = [hist_ttbar_score, hist_tttt_score] # List of histograms to draw

# --- Common Plot Customization ---
plot_pagerank.Title = "Example Page Rank Scores"
plot_pagerank.xTitle = "Page Rank Score (Arb.)"
plot_pagerank.yTitle = "Normalized Entries / Bin Width" # Adjusted title for density plot
plot_pagerank.xMin = 0
plot_pagerank.xMax = 1.0
plot_pagerank.xBins = 50             # Number of bins
plot_pagerank.xStep = 0.1            # Tick mark spacing on x-axis
plot_pagerank.yMin = 0               # Set y-axis minimum
# plot_pagerank.yMax = ...           # Optionally set y-axis maximum
plot_pagerank.Style = "ATLAS"        # Apply ATLAS plotting style
plot_pagerank.OutputDirectory = output_dir
plot_pagerank.Filename = "pagerank_example"
# plot_pagerank.ShowCount = True     # Display counts in the legend (useful for non-density)
plot_pagerank.Overflow = False       # Don't include overflow bin in normalization/display

# Save the figure
plot_pagerank.SaveFigure()
print(f"Saved PageRank plot to {output_dir}/{plot_pagerank.Filename}.png")


# 2. Stacked 1D Histogram (TH1F) - Invariant Mass

# Create histogram objects for each process
hist_ttbar_mass = TH1F()
hist_ttbar_mass.xData = ttbar_masses
hist_ttbar_mass.Color = "red"
hist_ttbar_mass.Title = r'$\bar{t}t$ Mass'

hist_tttt_mass = TH1F()
hist_tttt_mass.xData = tttt_masses
hist_tttt_mass.Color = "blue"
hist_tttt_mass.Title = r'$\bar{t}t\bar{t}t$ Mass'

# Create a histogram for the "truth" data (optional, drawn separately)
hist_truth_mass = TH1F()
hist_truth_mass.xData = truth_masses
hist_truth_mass.Color = "orange"
hist_truth_mass.Title = 'Truth Mass'
hist_truth_mass.LineStyle = "--" # Example: Make the line dashed

# Create the main plot object
plot_mass = TH1F()
plot_mass.Stacked = True             # Stack the histograms
plot_mass.Histograms = [hist_ttbar_mass, hist_tttt_mass] # Order matters for stacking
plot_mass.Histogram = hist_truth_mass # Add the truth histogram (drawn separately, not stacked)

# --- Customization ---
plot_mass.Title = "Example Invariant Mass Comparison"
plot_mass.xTitle = "Invariant Mass (GeV)"
plot_mass.yTitle = "Candidates / (4 GeV)" # Adjust based on bin width
plot_mass.xMin = 80
plot_mass.xMax = 300
plot_mass.xBins = 55                 # (300 - 80) / 4 = 55 bins
plot_mass.xStep = 20
plot_mass.yMin = 0
plot_mass.Style = "ATLAS"
plot_mass.OutputDirectory = output_dir
plot_mass.Filename = "top_mass_example"
plot_mass.ShowCount = True         # Show counts for stacked histograms

# Save the figure
plot_mass.SaveFigure()
print(f"Saved Mass plot to {output_dir}/{plot_mass.Filename}.png")


# 3. 2D Histogram (TH2F) - Score vs. Mass

# Create the 2D histogram object
plot_2d_ttbar = TH2F()

# Assign data (ensure xData and yData have the same length)
# Apply the score cut as in the example
valid_indices = [i for i, score in enumerate(ttbar_scores) if score < 0.95]
plot_2d_ttbar.xData = [ttbar_masses[i] for i in valid_indices]
plot_2d_ttbar.yData = [ttbar_scores[i] for i in valid_indices]

# --- Customization ---
plot_2d_ttbar.Title = "Score vs. Mass ($t\\bar{t}$ Example)"
plot_2d_ttbar.xTitle = "Invariant Mass (GeV)"
plot_2d_ttbar.yTitle = "Page Rank Score (Arb.)"

# X-axis settings
plot_2d_ttbar.xMin = 80
plot_2d_ttbar.xMax = 300
plot_2d_ttbar.xBins = 55 # Match the 1D plot binning if desired
plot_2d_ttbar.xStep = 20

# Y-axis settings
plot_2d_ttbar.yMin = 0
plot_2d_ttbar.yMax = 1.0
plot_2d_ttbar.yBins = 50
plot_2d_ttbar.yStep = 0.1

plot_2d_ttbar.Color = "magma"       # Colormap for the 2D histogram
plot_2d_ttbar.Style = "ATLAS"
plot_2d_ttbar.OutputDirectory = output_dir
plot_2d_ttbar.Filename = "top_mass_score_ttbar_example"
# plot_2d_ttbar.ShowCounts = False # Usually not shown on 2D plots
# plot_2d_ttbar.LogZ = True      # Use logarithmic color scale if needed

# Save the figure
plot_2d_ttbar.SaveFigure()
print(f"Saved 2D plot to {output_dir}/{plot_2d_ttbar.Filename}.png")

# --- Summary ---
# - Use TH1F for 1D histograms.
# - Use TH2F for 2D histograms.
# - Assign data using `.xData` (and `.yData` for TH2F).
# - Combine multiple 1D histograms using `.Histograms = [hist1, hist2, ...]`.
# - Set `.Stacked = True` for stacked 1D plots.
# - Set `.Density = True` for area-normalized 1D plots.
# - Customize titles, labels, ranges, bins, colors, style etc. using object attributes.
# - Call `.SaveFigure()` to generate the plot image.
# - Ensure the OutputDirectory exists or is created before saving.

import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# (Re-run the save commands if the directory was just created)
# plot_pagerank.SaveFigure()
# plot_mass.SaveFigure()
# plot_2d_ttbar.SaveFigure()

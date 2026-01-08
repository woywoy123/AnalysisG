import os
import sys
from typing import Dict, List, Tuple, Any
from AnalysisG.core.plotting import TH1F, TH2F
from AnalysisG import Analysis, EventGNN, TopEfficiency

# Tutorial: Complete Analysis Pipeline with AnalysisG
#
# This script demonstrates a full analysis workflow using the AnalysisG framework.
# It reconstructs top quarks from simulated ttbar and tttt events and compares results.


# --- 1. Import Necessary Modules ---
# Assuming 'figures.py' contains necessary helper functions or classes.
# If not used, remove this import.
# from figures import *

# --- 2. Configuration ---
# Input Data Paths (Replace <name> with the appropriate prefix)
# These ROOT files should contain graph representations of particle collision events.
DATA_TTBAR_PATH = "<name>GraphJets_bn_1_Grift/MRK-1/epoch-130/kfold-1/user.410465.aMcAtNloPy8EvtGen.DAOD_TOPQ1.e6762_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root/user.tnommens.40945548._000147.output.root"
DATA_4TOP_PATH = "<name>GraphJets_bn_1_Grift/MRK-1/epoch-130/kfold-1/user.412043.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e7101_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root/user.tnommens.40946130._000009.output.root"

# Processing Flags
RUN_ANALYSIS = False  # Set to True to re-run event processing, False to load existing results.
NUM_THREADS = 4       # Number of threads for parallel processing if RUN_ANALYSIS is True.
OUTPUT_DIR = "./output/" # Directory for plots and saved results
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

# --- 3. Helper Functions ---

def run_event_processing(data_path: str, sample_name: str, num_threads: int) -> TopEfficiency:
    """Runs the AnalysisG event processing loop."""
    print(f"--- Running AnalysisG event processing for '{sample_name}' sample ---")
    sel = TopEfficiency()
    ana = Analysis()
    ev = EventGNN()

    ana.Threads = num_threads
    ana.AddEvent(ev, "tmp") # "tmp" is a temporary identifier
    ana.AddSelection(sel)
    ana.AddSamples(data_path, "tmp")

    print("--- Starting AnalysisG ---")
    ana.Start()
    print("--- AnalysisG finished ---")

    output_file = os.path.join(OUTPUT_DIR, f"{sample_name}.pkl")
    print(f"--- Dumping results to '{output_file}' ---")
    sel.dump(output_file)
    print("--- Results dumped ---")
    return sel

def load_processed_results(sample_name: str) -> TopEfficiency:
    """Loads previously processed and saved results."""
    input_file = os.path.join(OUTPUT_DIR, f"{sample_name}.pkl")
    print(f"--- Loading pre-computed results from '{input_file}' ---")
    if not os.path.exists(input_file):
        print(f"ERROR: Results file not found: {input_file}")
        print("Set RUN_ANALYSIS = True and run the script first for both ttbar and tttt.")
        sys.exit(1)

    sel = TopEfficiency()
    loaded_data = sel.load(input_file)
    print(f"--- Loaded {sample_name} results ---")
    return loaded_data

def revert_mapping(data: TopEfficiency) -> Tuple[Dict[Any, List[float]], Dict[Any, List[float]]]:
    """Restructures loaded score and mass data from the TopEfficiency object."""
    pred_mass = data.p_topmass # Predicted top mass
    scr_mass = data.prob_tops   # Associated score (e.g., PageRank)
    mass_map: Dict[Any, List[float]] = {}
    score_map: Dict[Any, List[float]] = {}

    for event_key, event_mass_data in pred_mass.items():
        event_score_data = scr_mass.get(event_key, {})
        for cand_key, cand_masses in event_mass_data.items():
            if cand_key not in score_map:
                mass_map[cand_key] = []
                score_map[cand_key] = []
            mass_map[cand_key].extend(cand_masses)
            score_map[cand_key].extend(event_score_data.get(cand_key, []))
    return score_map, mass_map

# --- 4. Plotting Functions ---

def plot_score_distribution(ttbar_scores: List[float], tttt_scores: List[float], output_dir: str):
    """Generates and saves the PageRank score distribution plot."""
    print("--- Generating PageRank score plot ---")
    th_score_ttbar = TH1F(
        Title=r'$t\bar{t}$',
        xData=[s for s in ttbar_scores if s < 0.99],
        Color="red",
        Density=True,
        ErrorBars=True
    )
    th_score_tttt = TH1F(
        Title=r'$t\bar{t}t\bar{t}$',
        xData=[s for s in tttt_scores if s < 0.99],
        Color="blue",
        Density=True,
        ErrorBars=True
    )

    plot_scores = TH1F(
        Title="Page Rank Scores of Reconstructed Top Candidates",
        xTitle="Page Rank Score of Candidate Tops (Arb.)",
        yTitle="Normalized Density",
        xMin=0, xMax=1.01, xBins=200, xStep=0.1,
        Overflow=False,
        Style="ATLAS",
        OutputDirectory=output_dir,
        Filename="pagerank"
    )
    plot_scores.Histograms = [th_score_ttbar, th_score_tttt]
    plot_scores.SaveFigure()

def plot_mass_distribution(ttbar_masses: List[float], tttt_masses: List[float], truth_masses: List[float], output_dir: str):
    """Generates and saves the invariant mass distribution plot."""
    print("--- Generating invariant mass plot ---")
    th_mass_truth = TH1F(
        Title='Truth Tops',
        xData=truth_masses,
        Color="orange"
    )
    th_mass_ttbar = TH1F(
        Title=r'$t\bar{t}$',
        xData=ttbar_masses,
        Color="red"
    )
    th_mass_tttt = TH1F(
        Title=r'$t\bar{t}t\bar{t}$',
        xData=tttt_masses,
        Color="blue"
    )

    plot_mass = TH1F(
        Title="Unweighted Invariant Mass of Reconstructed Top Candidates",
        xTitle="Invariant Mass of Top Candidate (GeV)",
        yTitle="Number of Top Candidates / (1 GeV)", # Bin width = (300-80)/220 = 1 GeV
        xMin=80, xMax=300, xBins=220, xStep=20,
        Overflow=False,
        Stacked=True,
        Style="ATLAS",
        OutputDirectory=output_dir,
        Filename="top-mass"
    )
    # Set Histogram for potential base drawing, Histograms for stacking/overlay
    plot_mass.Histogram = th_mass_truth
    plot_mass.Histograms = [th_mass_ttbar, th_mass_tttt]
    plot_mass.SaveFigure()

def plot_mass_score_2d(masses: List[float], scores: List[float], sample_tag: str, title_tag: str, output_dir: str):
    """Generates and saves a 2D score vs. mass plot."""
    print(f"--- Generating 2D score vs mass plot ({sample_tag}) ---")
    # Filter data (ensure corresponding mass/score pairs are kept)
    indices = [i for i, score in enumerate(scores) if score < 0.95]
    filtered_masses = [masses[i] for i in indices]
    filtered_scores = [scores[i] for i in indices]

    plot_mass_score = TH2F(
        Title=f"PageRank Score as a Function of Top Candidate Invariant Mass ({title_tag})",
        xTitle="Invariant Mass of Top Candidate (GeV)",
        yTitle="Page Rank Score of Top Candidate (Arb.)",
        xMin=80, xMax=300, xBins=220, xStep=20,
        yMin=0, yMax=1.0, yBins=100, yStep=0.1,
        xData=filtered_masses,
        yData=filtered_scores,
        Color="magma",
        Style="ATLAS",
        OutputDirectory=output_dir,
        Filename=f"top-mass-score-{sample_tag}"
    )
    plot_mass_score.SaveFigure()

def create_plots(ttbar_data: TopEfficiency, tttt_data: TopEfficiency, output_dir: str):
    """Processes loaded data and generates all plots."""
    print("--- Preparing data for plotting ---")
    ttbar_score_map, ttbar_mass_map = revert_mapping(ttbar_data)
    tttt_score_map, tttt_mass_map = revert_mapping(tttt_data)

    # Flatten data for plotting
    ttbar_scores_flat = sum(ttbar_score_map.values(), [])
    tttt_scores_flat = sum(tttt_score_map.values(), [])
    ttbar_masses_flat = sum(ttbar_mass_map.values(), [])
    tttt_masses_flat = sum(tttt_mass_map.values(), [])

    # Aggregate truth masses
    truth_masses = sum(
        (sum(ttbar_data.t_topmass[i].values(), []) for i in ttbar_data.t_topmass), []
    )
    truth_masses += sum(
        (sum(tttt_data.t_topmass[i].values(), []) for i in tttt_data.t_topmass), []
    )

    # Generate plots
    plot_score_distribution(ttbar_scores_flat, tttt_scores_flat, output_dir)
    plot_mass_distribution(ttbar_masses_flat, tttt_masses_flat, truth_masses, output_dir)
    plot_mass_score_2d(ttbar_masses_flat, ttbar_scores_flat, "ttbar", r"$t\bar{t}$", output_dir)
    plot_mass_score_2d(tttt_masses_flat, tttt_scores_flat, "tttt", r"$t\bar{t}t\bar{t}$", output_dir)

    print("--- Plotting finished ---")

# --- 5. Main Execution ---

def main():
    """Main execution function."""
    if RUN_ANALYSIS:
        # Run processing for both samples if requested
        print("--- Running Analysis Stage ---")
        sel_ttbar = run_event_processing(DATA_TTBAR_PATH, "ttbar", NUM_THREADS)
        sel_tttt = run_event_processing(DATA_4TOP_PATH, "tttt", NUM_THREADS)
        print("--- Analysis Stage Complete ---")
        # Optional: Exit after running analysis if plotting isn't immediately needed
        # print("--- Exiting after running analysis ---")
        # sys.exit(0)
    else:
        # Load existing results
        print("--- Loading Analysis Results ---")
        sel_ttbar = load_processed_results("ttbar")
        sel_tttt = load_processed_results("tttt")
        print("--- Loading Complete ---")

    # Generate plots using the loaded or processed data
    print("--- Starting Plotting Stage ---")
    # Pass the TopEfficiency objects directly to the plotting function
    create_plots(sel_ttbar, sel_tttt, OUTPUT_DIR)
    print("--- Plotting Stage Complete ---")

    print("--- Script execution complete ---")

if __name__ == "__main__":
    main()

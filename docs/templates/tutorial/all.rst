.. _examples_analysis_runner:

Analysis Runner Example
=======================

This example demonstrates how to configure and run an analysis using the ``AnalysisG`` framework. It parses a YAML configuration file to set up the analysis, including specifying the dataset, event and graph implementations, models, and training parameters.

.. code-block:: python

    from AnalysisG.generators.analysis import Analysis
    from AnalysisG.graphs import bsm_4tops
    from AnalysisG.events.bsm_4tops.event_bsm_4tops import BSM4Tops
    from runner import samples_mc16
    import argparse
    import yaml

    # Parse configuration file
    parser = argparse.ArgumentParser(description="Configuration YAML file")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Initialize analysis
    ana = Analysis()
    ana.OutputPath = config["output-path"]

    # Set up graph and event implementations
    graph_impl = bsm_4tops.GraphTops()
    event_impl = BSM4Tops()

    # Add samples
    samples = samples_mc16.samples(config["sample-path"], "mc16")
    for sample_name in config["samples"]:
        ana.AddSamples(samples[sample_name], sample_name)
        ana.AddEvent(event_impl, sample_name)
        ana.AddGraph(graph_impl, sample_name)

    # Configure training parameters
    ana.Threads = config.get("threads", 1)
    ana.BatchSize = config.get("batch_size", 32)
    ana.Epochs = config.get("epochs", 10)

    # Start analysis
    ana.Start()

.. _examples_plotting_histograms:

Plotting Histograms Example
===========================

This example shows how to create and save histograms based on analysis results.

.. code-block:: python

    from AnalysisG.core.plotting import TH1F

    def create_histogram(title, data, bins, x_min, x_max, x_title):
        hist = TH1F()
        hist.Title = title
        hist.xData = data
        hist.xBins = bins
        hist.xMin = x_min
        hist.xMax = x_max
        hist.xTitle = x_title
        hist.yTitle = "Events"
        hist.SaveFigure()

    # Example usage
    data = [10, 20, 30, 40, 50]  # Example data
    create_histogram("Example Histogram", data, 10, 0, 100, "X-axis Label")

.. _examples_plotting_lines:

Plotting Line Graphs Example
============================

This example demonstrates how to create line plots for benchmarking results.

.. code-block:: python

    from AnalysisG.core.plotting import TLine

    def create_line_plot(title, x_data, y_data, x_title, y_title):
        line = TLine()
        line.Title = title
        line.xData = x_data
        line.yData = y_data
        line.xTitle = x_title
        line.yTitle = y_title
        line.SaveFigure()

    # Example usage
    x_data = [1, 2, 3, 4, 5]
    y_data = [10, 20, 30, 40, 50]
    create_line_plot("Example Line Plot", x_data, y_data, "X-axis", "Y-axis")

.. _examples_data_loading:

Data Loading Example
====================

This example shows how to load and process data from files.

.. code-block:: python

    import pickle

    def load_data(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data

    # Example usage
    data = load_data("example.pkl")
    print(data)

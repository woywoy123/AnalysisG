Selections
==========

Event selection algorithms for cut-based physics analyses.

Overview
--------

The ``selections`` module contains selection implementations for various studies:

- **analysis**: Main analysis selections with regions
- **example**: Example selections for tutorials
- **mc16**: MC16 truth studies
- **mc20**: MC20 truth studies  
- **neutrino**: Neutrino reconstruction studies
- **performance**: Performance evaluation selections

Selection Organization
----------------------

Selections are organized by:

1. **Physics Analysis**: Different final states (4-tops, ttbar, etc.)
2. **Selection Regions**: Signal, control, validation regions
3. **Study Type**: Truth matching, kinematics, reconstruction

MC16 Selections
---------------

Truth-level studies on MC16 samples:

- **childrenkinematics**: Top decay product kinematics
- **decaymodes**: Top quark decay mode classification
- **met**: Missing transverse energy studies
- **parton**: Parton-level analysis
- **topjets**: Top-tagged jet studies
- **topkinematics**: Top quark kinematic distributions
- **topmatching**: Truth matching algorithms
- **toptruthjets**: Truth jet associations
- **zprime**: Z' resonance searches

MC20 Selections
---------------

Studies on MC20 simulation samples:

- **matching**: Object matching algorithms
- **topkinematics**: Top kinematic distributions
- **topmatching**: Truth matching performance
- **zprime**: Heavy resonance searches

Neutrino Selections
-------------------

Neutrino reconstruction studies:

- **combinatorial**: Combinatorial reconstruction methods
- **validation**: Validation of reconstruction algorithms

Performance Selections
----------------------

Framework performance studies:

- **topefficiency**: Top reconstruction efficiency

Selection Template Usage
------------------------

All selections inherit from ``selection_template`` and implement:

1. **Selection()**: Define event selection logic
2. **InitHistograms()**: Book histograms
3. **ApplySelection()**: Apply cuts and fill histograms
4. **Finalize()**: Produce final plots and statistics

Cutflow Tracking
----------------

Selections automatically track:

- Number of events at each cut
- Efficiency of each cut
- Cumulative selection efficiency
- Statistical uncertainties

This enables:

- Optimization of cut ordering
- Understanding selection performance
- Debugging selection logic
- Reporting results

Output
------

Selections produce:

- **Histograms**: Kinematic distributions
- **Tables**: Cutflow statistics
- **ROOT Files**: For further analysis
- **Plots**: Publication-quality figures

Integration with Analysis
--------------------------

Selections are executed via the Analysis framework:

1. Events are loaded from ROOT files
2. Events are passed to Selection instances
3. Selections apply cuts and fill histograms
4. Results are aggregated across all events
5. Final outputs are produced

This allows:

- Processing large datasets efficiently
- Running multiple selections in parallel
- Caching intermediate results
- Modular analysis workflows

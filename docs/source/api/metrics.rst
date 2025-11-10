Metrics
=======

Performance metrics for model evaluation and physics analysis.

Overview
--------

The ``metrics`` module provides metrics for assessing model performance:

- **accuracy**: Classification accuracy metrics
- **pagerank**: Graph-based ranking metrics

Accuracy Metrics
----------------

The ``accuracy`` metric provides standard classification performance measures:

- Binary accuracy
- Multi-class accuracy
- Top-k accuracy
- Per-class accuracy

Usage:
- Accumulates predictions across batches
- Computes final accuracy statistics
- Supports weighted and unweighted modes

PageRank Metrics
----------------

The ``pagerank`` metric uses the PageRank algorithm for graph analysis:

- Node importance ranking
- Graph centrality measures
- Connection strength analysis

Usage:
- Evaluates graph structure quality
- Ranks nodes by importance
- Useful for understanding learned representations

Custom Metrics
--------------

Users can define custom metrics by:

1. Inheriting from ``metric_template``
2. Implementing accumulation logic
3. Implementing reduction logic
4. Implementing metric computation
5. Registering with the framework

Metric Usage
------------

Metrics are used during:

- **Training**: Monitor training progress
- **Validation**: Select best models
- **Testing**: Final performance evaluation
- **Analysis**: Understanding model behavior

Metrics can be:

- Computed per-batch
- Accumulated across batches
- Reduced across epochs
- Logged and visualized

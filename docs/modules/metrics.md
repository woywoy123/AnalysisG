# Metrics Module

@brief Tools for evaluating model performance and generating plots

## Overview

The Metrics module provides utilities for evaluating the performance of machine learning models and generating visualizations such as loss and accuracy plots.

## Key Components

### metrics

Class for managing metrics and generating plots.

```cpp
class metrics: public tools, public notification
{
    void dump_loss_plots(int k);
    void dump_accuracy_plots(int k);
    void dump_mass_plots(int k);
};
```

### model_report

Struct for storing and reporting model performance metrics.

```cpp
struct model_report {
    std::map<mode_enum, std::map<std::string, float>> loss_graph;
    std::map<mode_enum, std::map<std::string, float>> accuracy_graph;
};
```

## Usage Example

```cpp
// Create a metrics object
metrics* metric = new metrics();

// Register a model
model_report* report = metric->register_model(model, kfold);

// Generate plots
metric->dump_loss_plots(kfold);
metric->dump_accuracy_plots(kfold);
```

## Advanced Features

- **Custom Metrics**: Define and register custom metrics for specific use cases.
- **Visualization**: Generate detailed plots for loss, accuracy, and other metrics.
- **Integration with Models**: Seamlessly integrate with model templates for automated metric tracking.
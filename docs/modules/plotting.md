# Plotting Module

@brief Tools for data visualization and plotting

## Overview

The Plotting module provides utilities for creating and customizing plots. It supports a variety of plot types and customization options for scientific visualization.

## Key Components

### plotting

Class for managing and generating plots.

```cpp
class plotting: public tools, public notification
{
    // Class for managing and generating plots
    void build_ROC(std::string name, int kfold, std::vector<int>* labels, std::vector<std::vector<double>>* scores);
    void build_error();
};
```

## Usage Example

```cpp
// Create a plotting object
plotting* plot = new plotting();

// Generate a ROC curve
plot->build_ROC("ROC_Curve", kfold, labels, scores);

// Build error bars
plot->build_error();
```

## Advanced Features

- **Customization**: Customize plot styles, colors, and labels.
- **Scientific Visualization**: Generate plots for ROC curves, error bars, and more.
- **Integration with Metrics**: Seamlessly integrate with the Metrics module for automated plotting.
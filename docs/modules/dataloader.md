# Dataloader Module

@brief Data loading and batching framework

## Overview

The Dataloader module provides tools for loading, batching, and managing data for machine learning models. It supports efficient data handling and integration with graph-based data structures.

## Key Components

### dataloader

Class for managing data loading and batching.

```cpp
class dataloader: public notification, public tools
{
    std::vector<graph_t*>* build_batch(std::vector<graph_t*>* data, model_template* mdl, model_report* rep);
    void dump_dataset(std::string path);
};
```

## Usage Example

```cpp
// Create a dataloader object
dataloader* loader = new dataloader();

// Generate batches
std::vector<graph_t*>* batch = loader->build_batch(data, model, report);

// Dump dataset to file
loader->dump_dataset("output_path");
```

## Advanced Features

- **Batching**: Efficiently create batches of data for training and evaluation.
- **Dataset Management**: Load, dump, and restore datasets.
- **Integration with Graphs**: Seamless handling of graph-based data structures.
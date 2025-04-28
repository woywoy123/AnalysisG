# Quick Start Guide

This guide will help you quickly get started with the AnalysisG framework for implementing a basic physics analysis workflow.

## Prerequisites

Ensure you have AnalysisG installed as described in the [Installation Guide](installation.md).

## Creating a Simple Analysis

Below is a step-by-step guide to create a basic analysis using AnalysisG.

### Step 1: Set Up Your Project Structure

Create a new directory for your analysis:

```bash
mkdir my_first_analysis
cd my_first_analysis
```

### Step 2: Create a Configuration File

Create a JSON configuration file named `config.json`:

```json
{
  "run_name": "top_tagging",
  "output_path": "./output",
  
  "data": {
    "input_files": ["path/to/data1.root", "path/to/data2.root"],
    "tree_name": "Physics",
    "train_fraction": 0.7,
    "validation_fraction": 0.15,
    "test_fraction": 0.15
  },
  
  "model": {
    "name": "GNN",
    "layers": [64, 128, 64],
    "dropout": 0.2,
    "activation": "relu"
  },
  
  "training": {
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "epochs": 50
  },
  
  "features": {
    "node_features": ["pt", "eta", "phi", "energy"],
    "edge_features": ["deltaR"],
    "graph_features": ["total_pt", "njets"]
  }
}
```

### Step 3: Create Your Analysis Script

Create a C++ file named `run_analysis.cxx`:

```cpp
#include <AnalysisG/analysis.h>
#include <AnalysisG/tools.h>
#include <AnalysisG/structs.h>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    // Create and configure the analysis
    analysis* an = new analysis();
    
    // Load configuration from JSON file
    std::string config_file = (argc > 1) ? argv[1] : "config.json";
    settings_t settings;
    an->load_config(config_file, &settings);
    an->import_settings(&settings);
    
    // Define graph structure and features
    an->set_graph_template([](event_template* event) {
        graph_template* graph = new graph_template();
        
        // Build graph from event
        graph->build(event);
        
        // Add node features
        graph->add_node_data_feature<double>(particle_template::pt, "pt");
        graph->add_node_data_feature<double>(particle_template::eta, "eta");
        graph->add_node_data_feature<double>(particle_template::phi, "phi");
        graph->add_node_data_feature<double>(particle_template::energy, "energy");
        
        // Add truth features for training
        graph->add_node_truth_feature<int>(particle_template::is_signal, "is_signal");
        
        // Define graph topology (fully connected in this example)
        graph->define_topology(topology_type::fully_connected);
        
        return graph;
    });
    
    // Run the analysis
    std::cout << "Starting analysis..." << std::endl;
    an->run();
    
    // Generate performance plots
    an->generate_plots();
    
    std::cout << "Analysis complete. Results saved to: " << settings.output_path << std::endl;
    
    delete an;
    return 0;
}
```

### Step 4: Compile Your Analysis

Create a simple CMake file named `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.14)
project(my_first_analysis)

find_package(AnalysisG REQUIRED)

add_executable(run_analysis run_analysis.cxx)
target_link_libraries(run_analysis PRIVATE AnalysisG::Core)

set_target_properties(run_analysis PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
```

Compile the project:

```bash
mkdir build && cd build
cmake ..
make
```

### Step 5: Run Your Analysis

```bash
./run_analysis ../config.json
```

## What's Next?

After completing this quickstart guide, you can:

1. Explore more complex [graph topologies](user_guide/graphs.md)
2. Implement [custom selection criteria](user_guide/analysis.md#event-selection)
3. Try different [model architectures](user_guide/models.md)
4. Learn how to [visualize your results](user_guide/metrics.md#visualization)

For more detailed examples, check out the [Tutorials](tutorials/index.md) section.
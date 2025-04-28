# PyC Module

@brief Python-C++ integration for machine learning

## Overview

The PyC module provides tools for integrating Python-based machine learning models with the C++ framework. It enables seamless communication between Python and C++ components, allowing the use of PyTorch models in the analysis pipeline.

## Key Components

### pyc

Class for managing Python-C++ integration.

```cpp
class pyc
{
    void load_model(std::string model_path); // Loads a PyTorch model from a file
    torch::Tensor predict(torch::Tensor input); // Runs inference using the loaded model
    void set_device(std::string device); // Sets the computation device (CPU/GPU)
};
```

## Usage Example

```cpp
// Create a PyC object
pyc* py_interface = new pyc();

// Load a PyTorch model
py_interface->load_model("model.pt");

// Set the computation device
py_interface->set_device("cuda");

// Run inference
torch::Tensor input = torch::randn({1, 3, 224, 224});
torch::Tensor output = py_interface->predict(input);
```

## Advanced Features

- **Device Management**: Switch between CPU and GPU for computation.
- **Model Loading**: Load and manage PyTorch models dynamically.
- **Integration**: Seamlessly integrate Python-based models into the C++ analysis pipeline.
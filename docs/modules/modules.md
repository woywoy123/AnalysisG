# Modules Framework

@brief Modular framework for extending functionality

## Overview

The Modules framework provides a system for creating and managing modular components within the AnalysisG framework. It allows for the dynamic addition of new features and tools.

## Key Components

### module_template

Base class for creating custom modules.

```cpp
class module_template
{
    virtual void initialize(); // Initializes the module
    virtual void execute(); // Executes the module's functionality
    virtual void finalize(); // Finalizes the module
};
```

### module_manager

Class for managing multiple modules.

```cpp
class module_manager
{
    void add_module(module_template* module); // Adds a module to the manager
    void run_all(); // Runs all modules in sequence
    void clear_modules(); // Clears all modules from the manager
};
```

## Usage Example

```cpp
// Create a custom module
class custom_module : public module_template
{
    void initialize() override {
        std::cout << "Initializing custom module." << std::endl;
    }
    void execute() override {
        std::cout << "Executing custom module." << std::endl;
    }
    void finalize() override {
        std::cout << "Finalizing custom module." << std::endl;
    }
};

// Create a module manager
module_manager* manager = new module_manager();

// Add the custom module
custom_module* module = new custom_module();
manager->add_module(module);

// Run all modules
manager->run_all();
```

## Advanced Features

- **Dynamic Modules**: Add and manage modules dynamically at runtime.
- **Execution Control**: Control the execution order of modules.
- **Integration**: Seamlessly integrate custom modules into the framework.
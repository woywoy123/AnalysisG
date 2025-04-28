# Event Template Module

@brief Base class for event representation

## Overview

The Event Template module provides a base class for representing physics events. It defines the common interface and functionality for handling event data, including particle registration and data processing.

## Key Components

### event_template

Base class for event representations.

```cpp
class event_template
{
    // ...existing code...
    virtual void CompileEvent(); // Compiles the event data
    virtual bool PreSelection(); // Applies preselection criteria
    void register_particles(std::map<std::string, particle_type*>* container); // Registers particles
    void add_leaf(std::string key, std::string leaf_name); // Adds a data leaf
    void process(element_t* el); // Processes event data
    // ...existing code...
};
```

## Usage Example

```cpp
// Create an event object
event_template* event = new event_template();

// Register particles
std::map<std::string, particle_type*> particles;
event->register_particles(&particles);

// Add a data leaf
event->add_leaf("energy", "energy_leaf");

// Process event data
element_t* element = /* data source */;
event->process(element);

// Apply preselection criteria
if (event->PreSelection()) {
    std::cout << "Event passed preselection." << std::endl;
}
```

## Advanced Features

- **Particle Registration**: Manage and register particles for event analysis.
- **Data Processing**: Add and process data leaves dynamically.
- **Preselection**: Apply custom preselection criteria to filter events.
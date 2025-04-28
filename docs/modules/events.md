# Events Module

@brief Event representation and processing framework

## Overview

The Events module provides tools for representing and processing physics events. It includes classes for managing event data, applying transformations, and interfacing with other modules.

## Key Components

### event_template

Base class for event representations.

```cpp
class event_template
{
    virtual void CompileEvent(); // Compiles the event data
    virtual bool PreSelection(); // Applies preselection criteria
    void register_particles(std::map<std::string, particle_type*>* container); // Registers particles
    void add_leaf(std::string key, std::string leaf_name); // Adds a data leaf
    void process(element_t* el); // Processes event data
};
```

### ssml_mc20::event

A concrete implementation for SSML MC20 format events.

```cpp
class event : public event_template
{
    void build(element_t* el) override;
    void CompileEvent() override;
    bool PreSelection() override;
};
```

## Usage Example

```cpp
// Create an event object
event_template* event = new ssml_mc20::event();

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
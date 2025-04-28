# Selection Module

@brief Event selection and filtering framework

## Overview

The Selection module provides a flexible system for defining and applying event selection criteria. It supports complex selection logic and efficiency tracking.

## Key Components

### selection_template

Base class for event selection implementations.

```cpp
class selection_template
{
    virtual bool selection(event_template* ev) = 0;
    virtual bool strategy(event_template* ev) = 0;
};
```

### topkinematics

Example implementation for top quark kinematics selection.

```cpp
class topkinematics: public selection_template
{
    bool selection(event_template* ev) override;
    bool strategy(event_template* ev) override;
};
```

## Usage Example

```cpp
// Create a selection object
selection_template* sel = new topkinematics();

// Apply selection to an event
if (sel->selection(event)) {
    // Event passed the selection
}
```

## Advanced Features

- **Custom Selections**: Define custom selection criteria for specific use cases.
- **Efficiency Tracking**: Track the efficiency of each selection criterion.
- **Integration with Events**: Seamlessly integrate with event templates for data access.
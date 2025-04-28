# Selections Module

@brief Event selection and filtering framework

## Overview

The Selections module provides tools for defining and applying event selection criteria. It supports complex selection logic and efficiency tracking.

## Key Components

### selection_template

Base class for event selection implementations.

```cpp
class selection_template
{
public:
    // Selection interface
    virtual bool apply_selection(event_template* event) = 0; // Applies the selection criteria
    
    // Metadata and tracking
    std::string get_name() const; // Retrieves the name of the selection
    float get_efficiency() const;
    int get_passed_events() const;
    int get_total_events() const;
    
    // Selection configuration
    void set_name(const std::string& name);
    void set_invert(bool invert);
    
protected:
    std::string name;
    bool invert_result = false;
    int total_events = 0;
    int passed_events = 0;
};
```

### selection_chain

Class for combining multiple selection criteria into a chain with configurable logic.

```cpp
class selection_chain : public selection_template
{
public:
    // Chain management
    void add_selection(selection_template* selection);
    void remove_selection(const std::string& name);
    
    // Logic configuration
    enum class Logic { AND, OR, XOR, NAND, NOR };
    void set_logic(Logic logic);
    
    // Apply the chain
    bool apply(event_template* evt) override;
    
    // Chain inspection
    std::vector<selection_template*> get_selections() const;
    selection_template* get_selection(const std::string& name) const;
    
private:
    std::vector<selection_template*> selections;
    Logic chain_logic = Logic::AND;
};
```

### selection_manager

Class for managing multiple selections.

```cpp
class selection_manager
{
public:
    // Selection management
    void add_selection(selection_template* selection); // Adds a selection to the manager
    bool apply_all(event_template* event); // Applies all selections to an event
    void clear_selections(); // Clears all selections from the manager
};
```

### Concrete Selections

Several implementations for common physics selection criteria:

```cpp
// Particle multiplicity selection
class multiplicity_selection : public selection_template
{
public:
    multiplicity_selection(const std::string& particle_type, int min_count, int max_count = -1);
    bool apply(event_template* evt) override;
    
private:
    std::string particle_type;
    int min_count;
    int max_count;
};

// Kinematic variable threshold selection
class kinematic_selection : public selection_template
{
public:
    enum class Variable { PT, ETA, PHI, E, MASS };
    enum class Comparison { GREATER, LESS, EQUAL, NOT_EQUAL };
    
    kinematic_selection(const std::string& particle_type, Variable var, 
                       Comparison comp, double threshold);
    bool apply(event_template* evt) override;
    
private:
    std::string particle_type;
    Variable variable;
    Comparison comparison;
    double threshold;
};

// ML-based selection
class ml_selection : public selection_template
{
public:
    ml_selection(model_template* model, double threshold);
    bool apply(event_template* evt) override;
    
private:
    model_template* model;
    double threshold;
};
```

## Usage Example

```cpp
// Create a selection object
selection_template* selection = new custom_selection();

// Create a selection manager
selection_manager* manager = new selection_manager();
manager->add_selection(selection);

// Apply selections to an event
if (manager->apply_all(event)) {
    std::cout << "Event passed all selections." << std::endl;
}

// Create individual selection criteria
auto electron_pt = new kinematic_selection("electron", 
                                          kinematic_selection::Variable::PT,
                                          kinematic_selection::Comparison::GREATER, 
                                          25.0);
electron_pt->set_name("ElectronPt");

auto muon_multiplicity = new multiplicity_selection("muon", 1);
muon_multiplicity->set_name("MuonMultiplicity");

// Create a selection chain with AND logic
auto selection_chain = new selection_chain();
selection_chain->set_name("ElectronMuonSelection");
selection_chain->set_logic(selection_chain::Logic::AND);
selection_chain->add_selection(electron_pt);
selection_chain->add_selection(muon_multiplicity);

// Create ML-based selection (optional additional layer)
auto ml_model = load_trained_model("models/event_classifier.pt");
auto ml_select = new ml_selection(ml_model, 0.8); // 80% confidence threshold
ml_select->set_name("MLSelection");

// Apply selection to events
for (auto& evt : events) {
    if (selection_chain->apply(evt)) {
        // Event passed the basic selection
        if (ml_select->apply(evt)) {
            // Event also passed the ML selection
            selected_events.push_back(evt);
        }
    }
}

// Get selection efficiencies
std::cout << "Basic selection efficiency: " << selection_chain->get_efficiency() << std::endl;
std::cout << "ML selection efficiency: " << ml_select->get_efficiency() << std::endl;
```

## Advanced Features

- **Custom Selections**: Implement custom selection criteria by extending the selection_template class.
- **Selection Management**: Manage and apply multiple selections using the selection manager.
- **Efficiency Tracking**: Track the efficiency of each selection criterion.
- **Efficiency Maps**: Generate N-dimensional efficiency maps for selections
- **Correlation Studies**: Tools for studying correlations between different selection criteria
- **Selection Optimization**: Methods for optimizing selection thresholds based on metrics
- **Background Rejection**: Tools for maximizing signal-to-background ratio
- **Selection Visualization**: Utilities for visualizing the impact of selections on distributions
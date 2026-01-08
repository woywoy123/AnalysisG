---
title: Event Template
---

# Event Template

### Event Template Source Files

Implementing a custom event requires creating two source files: a header file (`event.h`) and a source file (`event.cxx`). These files define the structure and behavior of your event class, inheriting from the base `event_template`.

**Header File (`event.h`)**

The header file declares the event class. It specifies the data members (variables and particle collections) and the methods that define the event's interaction with the analysis framework.

**Example: `event.h`**

```cpp
#ifndef EVENTS_EVENTNAME_H
#define EVENTS_EVENTNAME_H

// Required base class header
#include <templates/event_template.h>
// Include definitions for custom particle types used
#include "particles.h"

// Declare the custom event class, inheriting from event_template
class event_name : public event_template
{
public:
    // Constructor: Used for initialization and registration
    event_name();
    // Virtual destructor override (important for proper cleanup)
    ~event_name() override;

    // --- Public Data Members ---
    // Collections to hold particles after processing in CompileEvent
    std::vector<custom_particle*> some_objects = {};
    std::vector<particle_template*> some_particles = {};

    // Example event-level variable
    float some_variable = 0;

    // --- Overridden Framework Methods ---
    // Creates a new instance of this event type (required by the framework)
    event_template* clone() override;

    // Populates event data members from ROOT files using element_t
    void build(element_t* el) override;

    // Optional method for post-processing steps after particle building
    void CompileEvent() override;

private:
    // --- Private Data Members ---
    // Maps to store particles fetched by the framework during the build phase.
    // Keys are unique identifiers (hashes).
    std::map<std::string, custom_particle*> m_some_objects = {};
    std::map<std::string, custom_particle_v2*> m_some_particles = {};
};

#endif // EVENTS_EVENTNAME_H
```

**Source File (`event.cxx`)**

The source file provides the implementation for the methods declared in the header file. This is where the specific logic for reading data, registering components, and processing the event resides.

Key Method Implementations:

*   **Constructor (`event_name::event_name`)**:
    *   Sets the event's `name`.
    *   Uses `add_leaf()` to specify which variables (leaves) to read from the ROOT n-tuple and optionally map them to shorter keys. The first argument is the key used within the `build` method, and the second is the actual leaf name in the ROOT file.
    *   Specifies the required ROOT `trees` to be accessed.
    *   Calls `register_particles()` for each private particle map (`m_some_objects`, `m_some_particles`). This informs the framework which particle types to instantiate and manage for this event. The framework uses these maps to store the particles it builds.

*   **Destructor (`event_name::~event_name`)**:
    *   Provides a place for custom cleanup, although often empty if standard containers and framework management are sufficient. Overriding the virtual destructor of the base class is crucial for correct polymorphism behavior.

*   **`clone()`**:
    *   Returns a new instance of the `event_name` class, cast to the base `event_template*`. This is used internally by the framework, for example, during parallel processing setup.

*   **`build(element_t* el)`**:
    *   This method is called for each event read from the input files.
    *   It receives an `element_t*` object (`el`), which acts as a temporary container for the data requested via `add_leaf`.
    *   Uses `el->get("key", &variable)` to retrieve data associated with a registered `key` and assign it to the corresponding event member variable (e.g., `this->some_variable`). The `get` method automatically handles type casting based on the type of the provided variable pointer.

*   **`CompileEvent()`**:
    *   An optional method called after the framework has populated the registered particle maps (`m_some_objects`, `m_some_particles`).
    *   Used for event-level compilation or post-processing steps that require access to the fully built particles.
    *   Common uses include:
       *   Transferring particles from the private framework-managed maps to the public vectors (e.g., `this->some_objects`, `this->some_particles`) for easier access in analysis tasks.
       *   Performing truth matching between different particle collections.
       *   Calculating derived event quantities based on the final particle collections.

**Example: `event.cxx`**

```cpp
#include "event.h" // Include the header file

event_name::event_name() {
    // Assign a unique name to this event implementation
    this->name = "event_name";

    // Register leaves to be read from the ROOT file
    // Format: add_leaf("internal_key", "ROOT_leaf_name");
    this->add_leaf("some_variable", "some_very_long_variable_name_in_root");

    // Specify the ROOT TTree(s) containing the required leaves/branches
    this->trees = {"some-tree"};

    // Register the private particle maps with the framework
    // The framework will populate these maps during event building.
    this->register_particles(&this->m_some_objects);
    this->register_particles(&this->m_some_particles);
}

// Destructor implementation (can be empty if no custom cleanup needed)
event_name::~event_name() {}

// Clone implementation: returns a new instance of this event class
event_template* event_name::clone() {
    return (event_template*)new event_name();
}

// Build implementation: extracts data from element_t
void event_name::build(element_t* el) {
    // Use the internal key to retrieve the data and assign it
    el->get("some_variable", &this->some_variable);

    // --- Note ---
    // The framework automatically handles populating the registered
    // particle maps (m_some_objects, m_some_particles) based on
    // particle definitions and branch registrations elsewhere.
    // You typically don't interact directly with particle branches here.
}

// CompileEvent implementation: post-processing after particle building
void event_name::CompileEvent() {
    // Example: Populate public vectors from private maps

    // Process custom_particle objects
    for (auto const& [key, val] : this->m_some_objects) {
        this->some_objects.push_back(val);
    }

    // Process custom_particle_v2 objects and cast to base class pointer
    for (auto const& [key, val] : this->m_some_particles) {
        // Cast to particle_template* if needed for a common container type
        this->some_particles.push_back(static_cast<particle_template*>(val));
    }

    // Other potential operations: truth matching, calculating derived variables, etc.
}
```

#### Understanding `element_t`

The `element_t` struct serves as a temporary data carrier for a single event during the `build` phase.

*   **Data Storage**: It holds the values read from the ROOT file for the leaves requested via `add_leaf`.
*   **`get()` Method**: The primary way to interact with `element_t` is through its `get(key, pointer_to_variable)` method.
    *   It retrieves the data associated with the specified `key` (the internal key defined in `add_leaf`).
    *   Crucially, it automatically deduces the required data type from the type of the `pointer_to_variable` provided and performs the necessary casting. This simplifies data retrieval, even for complex types like nested vectors (e.g., `std::vector<std::vector<float>>`).
    *   For example, to request a `std::vector<std::vector<float>>` value from `element_t` is as simple as:
       ```cpp
         // define the type
         std::vector<std::vector<float>> some_variable;

         // use element_t (here called el, following from the above example)
         el -> get("some-varible", &some_variable);
       ```
*   **Key Mapping**: Using different internal keys in `add_leaf` than the actual ROOT leaf names allows for shorter, more convenient names within the `build` method, especially useful for very long or complex ROOT variable names.

#### Particle Registration and Management

The framework manages the lifecycle of particle objects associated with an event.

*   **Private Maps**: You declare private maps (e.g., `m_some_objects`) in your event header. The keys are `std::string` (unique particle identifiers generated by the framework, often hashes), and the values are pointers to the specific particle types.
*   **`register_particles()`**: Calling this method in the event constructor tells the framework:
    1.  Which particle types are associated with this event.
    2.  Which map should be used to store instances of that particle type for the current event.
*   **Framework Responsibility**: During event processing, the framework reads the necessary particle data (based on separate particle definitions, not shown here) and populates these registered private maps. It also handles the memory management (creation and deletion) of these particle objects.
*   **`CompileEvent` Usage**: The `CompileEvent` method is typically where you access the particles populated by the framework in the private maps and potentially copy pointers to them into public vectors for easier use in subsequent analysis steps.

For more detailed information about the base class methods and attributes, refer to the core class documentation for `event-template`.

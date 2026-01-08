Tutorial: Integrating a Custom C++ Particle with Python/Cython via CMake
=========================================================================

This tutorial guides you through defining a new particle type in C++ and making it accessible within the AnalysisG framework in Python using Cython bindings. The build process is managed by CMake.

We'll use placeholders like `<particle-name>` throughout this guide. You should replace these with names relevant to your specific particle (e.g., `Jet`, `Muon`).

1. Define the C++ Particle Class Header (`<particle-name>.h`)
-------------------------------------------------------------

First, create the header file for your C++ particle class. This file declares the class structure and its members.

*   **Location:** Typically `include/<particle-module>/<particle-name>.h`
*   **Content:**
    *   Use standard include guards (`#ifndef`, `#define`, `#endif`).
    *   Include the base particle template header: `#include "templates/particle_template.h"`.
    *   Declare your class, inheriting publicly from `particle_template`.
    *   Declare the constructor, destructor (override), specific member variables, and override the required virtual methods (`clone`, `build`).

```cpp
// include/<particle-module>/<particle-name>.h
#ifndef ANALYSISG_<PARTICLE_MODULE>_<PARTICLE_NAME>_H
#define ANALYSISG_<PARTICLE_MODULE>_<PARTICLE_NAME>_H

#include "templates/particle_template.h" // Base class
#include <string>
#include <vector>
#include <map>

// Forward declaration if needed
class element_t;

class <particle-name> : public particle_template {
public:
    // Constructor
    <particle-name>();

    // Destructor (must be virtual and override)
    ~<particle-name>() override;

    // --- Particle Specific Member Variables ---
    float key_variable; // Example: Replace with actual variables

    // --- Overridden Virtual Methods ---
    // Creates a copy of this particle type
    particle_template* clone() override;

    // Builds particle instances from raw data
    void build(std::map<std::string, particle_template*>* prt, element_t* el) override;

};

#endif // ANALYSISG_<PARTICLE_MODULE>_<PARTICLE_NAME>_H
```

2. Implement the C++ Particle Class (`<particle-name>.cxx`)
-----------------------------------------------------------

Next, implement the methods declared in the header file.

*   **Location:** Typically `src/<particle-module>/<particle-name>.cxx`
*   **Content:**
    *   Include the corresponding header (`#include "<particle-module>/<particle-name>.h"`).
    *   **Constructor:** Initialize members, set the unique `this->type` string (e.g., `"jet"`), register data leaves using `add_leaf()`, and call `apply_type_prefix()`.
    *   **Destructor:** Add cleanup code if necessary (often empty).
    *   **`clone()`:** Return a new instance: `return (particle_template*)new <particle-name>();`.
    *   **`build()`:** This method populates the event's particle map. Retrieve data arrays from the `element_t` object, loop through them, create new `<particle-name>` instances, populate their members, and add them to the `prt` map using their hash as the key.

```cpp
// src/<particle-module>/<particle-name>.cxx
#include "<particle-module>/<particle-name>.h"
#include "core/element_t.h" // Include if using element_t

// Constructor Implementation
<particle-name>::<particle-name>() {
    this->type = "<some-particle>"; // Unique type identifier string
    // Register data leaves (kinematics, properties)
    this->add_leaf("<kinematic-key>", "<leaf-name-of-kinematic>");
    // ... add other leaves ...
    this->apply_type_prefix(); // Prepend type to leaf names
    // Initialize member variables if needed
    this->key_variable = 0.0f;
}

// Destructor Implementation
<particle-name>::~<particle-name>() {}

// clone() Implementation
particle_template* <particle-name>::clone() {
    return (particle_template*)new <particle-name>();
}

// build() Implementation
void <particle-name>::build(std::map<std::string, particle_template*>* prt, element_t* el) {
    // Example: Retrieve a data array for 'key_variable'
    std::vector<float>* some_kinematic = nullptr;
    if (!el->get("<kinematic-key>", &some_kinematic)) return; // Check if data exists

    size_t n_particles = some_kinematic->size();
    for (size_t i = 0; i < n_particles; ++i) {
        <particle-name>* p = new <particle-name>();
        // Populate the particle's data
        p->key_variable = (*some_kinematic)[i];
        // ... populate other members ...

        // Add the particle to the event map using its hash
        (*prt)[std::string(p->hash)] = p;
    }
}
```

3. Declare C++ Interface for Cython (`__init__.pxd`)
----------------------------------------------------

Create a Cython definition file (`.pxd`) to expose the C++ class structure to Cython.

*   **Location:** Typically `AnalysisG/events/<particle-name>/__init__.pxd`
*   **Content:**
    *   Use `cdef extern from` to point to the C++ header.
    *   Declare the `cppclass` mirroring the C++ class, including the base class and any members you need to access directly from Cython. Mark the constructor with `except+` if it can throw exceptions.

```cython
# AnalysisG/events/<particle-name>/__init__.pxd
from AnalysisG.core.particle_template cimport particle_template # Import C++ base class definition

cdef extern from "AnalysisG/particles/<particle-name>.h" namespace "AnalysisG": # Adjust path/namespace if needed
    cdef cppclass <particle-name>(particle_template):
        # Declare constructor (mark if it can throw exceptions)
        <particle-name>() except +
        # Declare C++ members needed in Cython
        float key_variable

# Optional: Declare the Python wrapper class if defined in a separate .pyx
# from AnalysisG.core.particle_template cimport ParticleTemplate # Import Python base class
# cdef class <Python-Particle>(ParticleTemplate):
#     pass
```

4. Define the Python Wrapper Class (`<Python-Particle>.pyx`)
------------------------------------------------------------

Create the Cython implementation file (`.pyx`) which defines the Python class that wraps the C++ object.

*   **Location:** Typically `AnalysisG/events/<particle-name>/<Python-Particle>.pyx` (or within `__init__.pyx`)
*   **Content:**
    *   Add compiler directives (`# distutils: language=c++`).
    *   `cimport` the C++ class definition from the `.pxd` file.
    *   Import the Python base class (`ParticleTemplate`).
    *   Define a `cdef class` inheriting from `ParticleTemplate`.
    *   Implement `__cinit__` to allocate the C++ object (`self.ptr = new <particle-name>()`).
    *   Implement `__dealloc__` to delete the C++ object (`del self.ptr`).
    *   Add Python properties or methods to expose C++ functionality as needed.

```cython
# AnalysisG/events/<particle-name>/<Python-Particle>.pyx
# distutils: language = c++
# cython: language_level=3

# Import the C++ class definition
from AnalysisG.events.<particle-name> cimport <particle-name>
# Import the Python base class wrapper
from AnalysisG.core.particle_template cimport ParticleTemplate

cdef class <Python-Particle>(ParticleTemplate):

    def __cinit__(self):
        # Allocate the C++ object when the Python object is created
        self.ptr = new <particle-name>()

    # __init__ is for Python-level initialization (often not needed here)
    # def __init__(self):
    #     pass

    def __dealloc__(self):
        # Deallocate the C++ object when the Python object is garbage collected
        if self.ptr != NULL:
            del self.ptr
            self.ptr = NULL # Good practice to nullify after deletion

    # --- Example: Expose C++ member as a Python property ---
    @property
    def KeyVariable(self):
        # Cast self.ptr to the C++ type to access members
        return (<<particle-name>*>self.ptr).key_variable

    @KeyVariable.setter
    def KeyVariable(self, float value):
        (<<particle-name>*>self.ptr).key_variable = value

    # Add other methods/properties as needed
```

5. Configure CMake Build (`CMakeLists.txt`)
-------------------------------------------

Finally, update your `CMakeLists.txt` to build the C++ code as a static library and then build the Cython extension, linking against the C++ library.

*   **Location:** The `CMakeLists.txt` managing your C++ source and Cython extensions.
*   **Content:**
    *   Define variables for your particle's source and header files.
    *   Use `add_library` to create a static library (e.g., `c<particle-name>`) from the C++ source.
    *   Use `target_include_directories` to specify header locations (both private for the particle itself and public for base classes).
    *   Use `target_link_libraries` to link against dependencies (like the base `cparticle_template` library).
    *   Set compile options like `-fPIC`.
    *   Use a CMake function (like the custom `cybuild` often used in AnalysisG projects) to compile the Cython `.pyx` file into a Python extension module (`.so`/`.pyd`), ensuring it links against your `c<particle-name>` static library.
    *   Use `install` commands to place the compiled Python module and the `.pxd`/`.py` interface files into the correct locations within the installed Python package.

```cmake
# CMakeLists.txt (Example Snippets)

# --- Define C++ Particle Library ---
set(PARTICLE_NAME "<particle-name>") # e.g., Jet
set(PARTICLE_MODULE "<particle-module>") # e.g., particles
set(CPP_HEADER_FILES include/${PARTICLE_MODULE}/${PARTICLE_NAME}.h)
set(CPP_SOURCE_FILES src/${PARTICLE_MODULE}/${PARTICLE_NAME}.cxx)

# Create static library for the C++ particle code
add_library(c${PARTICLE_NAME} STATIC ${CPP_SOURCE_FILES})

# Include directories for the C++ library
target_include_directories(c${PARTICLE_NAME}
    PRIVATE include/${PARTICLE_MODULE} # Particle's own header dir
    PUBLIC include                     # Base class headers dir
)

# Link C++ library against base particle template library
target_link_libraries(c${PARTICLE_NAME} PUBLIC cparticle_template) # Adjust dependency name if needed

# Set compile options (Position Independent Code is crucial for linking into .so)
target_compile_options(c${PARTICLE_NAME} PRIVATE -fPIC)

# --- Build Cython Extension ---
# Assuming a custom function 'cybuild' exists:
# cybuild(<Python Module Path> <Source Directory> <Output Module Name> <C++ Libs to Link>)
cmake_language(
    CALL cybuild
    "events/${PARTICLE_NAME}"                 # Python path (e.g., AnalysisG.events.Jet)
    "AnalysisG/events/${PARTICLE_NAME}"       # Directory containing .pyx/.pxd
    "${PARTICLE_NAME}"                        # Output module name (e.g., Jet.so)
    "c${PARTICLE_NAME}"                       # Link against our C++ library
)

# --- Installation ---
# Install the compiled Python extension module
install(TARGETS ${PARTICLE_NAME}
    DESTINATION ${SKBUILD_PROJECT_NAME}/events/${PARTICLE_NAME} # Adjust path based on skbuild/setup.py
)

# Install Cython interface files (.pxd) and Python init file (.py)
# Assuming they are in AnalysisG/events/<particle-name>/
install(FILES AnalysisG/events/${PARTICLE_NAME}/__init__.pxd
        DESTINATION ${SKBUILD_PROJECT_NAME}/events/${PARTICLE_NAME}
)
install(FILES AnalysisG/events/${PARTICLE_NAME}/__init__.py # Create this if needed
        DESTINATION ${SKBUILD_PROJECT_NAME}/events/${PARTICLE_NAME}
)
```

Placeholders Summary
--------------------

Remember to replace these placeholders:

*   `<particle-module>`: Name of the C++ module/directory (e.g., `particles`).
*   `<particle-name>`: Name of the C++ particle class (e.g., `Jet`, `Muon`). Use consistent capitalization.
*   `<Python-Particle>`: Name of the Python wrapper class (e.g., `CyJet`, `CyMuon`). Often the same as `<particle-name>`.
*   `<some-particle>`: String identifier for the particle type used in `this->type` (e.g., `"jet"`, `"muon"`).
*   `<kinematic-key>`: Key used to retrieve data arrays from `element_t` (e.g., `"jet_pt"`, `"muon_eta"`).
*   `<leaf-name-of-kinematic>`: Name assigned to the data leaf via `add_leaf` (e.g., `"pt"`, `"eta"`).

After completing these steps and rebuilding your project, you should be able to import and use your new particle type (`<Python-Particle>`) from Python within the AnalysisG framework.

.. _tutorial-custom-event:

Tutorial: Creating a Custom Event Class for AnalysisG
======================================================

This tutorial guides you through creating a basic C++ event class, wrapping it with Cython for Python integration, and configuring the build process using CMake within the AnalysisG framework. An "event" typically represents a collision event, holding event-level information.

1. File Structure
-----------------

Organize your files as follows, replacing ``<module>`` and ``<name>`` (e.g., ``MyEvents``, ``MyEvent``):

*   **Header:** ``include/<module>/<name>.h`` (C++ class declaration)
*   **Source:** ``cxx/<name>.cxx`` (C++ class implementation)
*   **Cython Definition:** ``events/<module>/<name>.pxd`` (Cython C++ declaration)
*   **Cython Implementation:** ``events/<module>/<name>.pyx`` (Python wrapper class)
*   **CMake:** Add rules to ``CMakeLists.txt``

2. C++ Header (``include/<module>/<name>.h``)
---------------------------------------------

Declare your C++ event class, inheriting from ``event_template``.

.. code-block:: cpp

    #ifndef EVENTS_<NAME>_H // Use uppercase name for guard
    #define EVENTS_<NAME>_H

    #include <templates/event_template.h> // Base event class
    #include <core/element_t.h>         // For build method parameter

    #include <string> // Include necessary standard types

    class <name> : public event_template {
    public:
        // --- Constructor & Destructor ---
        <name>();
        ~<name>() override; // Must override virtual destructor

        // --- User Data ---
        // Example: An event-level variable (e.g., event weight)
        float event_weight = 0;

        // --- Required Overrides ---
        event_template* clone() override; // Creates a copy
        void build(element_t* el) override; // Reads data from input

        // Optional override for post-processing (can be added later)
        // void CompileEvent() override;
    };

    #endif // EVENTS_<NAME>_H

*   **Placeholders:** Replace ``<NAME>`` (uppercase) and ``<name>`` (lowercase).
*   **Inheritance:** Must inherit publicly from ``event_template``.
*   **Overrides:** You *must* implement ``clone()``, ``build()``, and the destructor.
*   **Data:** Add public members (like ``event_weight``) to store event data read from the input file.

3. C++ Source (``cxx/<name>.cxx``)
----------------------------------

Implement the methods declared in the header.

.. code-block:: cpp

    #include "<name>.h" // Include the header for this class

    // --- Constructor ---
    <name>::<name>() {
        // Set a unique name for this event type
        this->name = "<name>";

        // --- Data Loading Configuration ---
        // Map a key (used in build()) to a leaf name in the input ROOT TTree
        this->add_leaf("event_weight_key", "mcEventWeight"); // Example mapping

        // Specify which TTree(s) contain the necessary data
        this->trees = {"Nominal"}; // Example tree name

        // Note: Particle handling (register_particle) is added here
        // if your event manages particle collections.
    }

    // --- Destructor ---
    // Usually empty; cleanup handled by base class or smart pointers.
    <name>::~<name>() {}

    // --- clone Method ---
    // Returns a new instance of this event class. Required by the framework.
    event_template* <name>::clone() {
        return (event_template*)new <name>();
    }

    // --- build Method ---
    // Extracts data from the input file for the current event using element_t.
    void <name>::build(element_t* el) {
        // Use the key defined in add_leaf() to retrieve the value
        el->get("event_weight_key", &this->event_weight);
        // Add more el->get(...) calls for other registered leaves
    }

    // --- CompileEvent Method (Optional) ---
    // Implement if needed for post-processing after particles are built.
    // void <name>::CompileEvent() { /* ... */ }

*   **Placeholders:** Replace ``<name>``. Ensure the key in ``add_leaf`` matches the key in ``el->get``. The second argument to ``add_leaf`` is the branch/leaf name in your input ROOT file.
*   **Constructor:** Configure data loading (``add_leaf``, ``trees``).
*   **``build``:** Use ``el->get("key", &member_variable)`` to read data.

4. Cython Definition (``events/<module>/<name>.pxd``)
----------------------------------------------------

Expose the C++ class structure to Cython.

.. code-block:: python

    # distutils: language=c++
    # cython: language_level=3

    from libcpp cimport bool # Import necessary C++ types if needed

    # Import base C++ and Python event templates
    from AnalysisG.core.event_template cimport event_template, EventTemplate

    # Declare the C++ class interface
    cdef extern from "<module>/<name>.h": # Path to your C++ header
        cdef cppclass <name>(event_template):
            # Declare constructor
            <name>() except + # Enable C++ exception handling

            # Declare public members to expose
            float event_weight
            # Add other public members from C++ header here

    # Declare the Python wrapper class type
    cdef class Py<name>(EventTemplate): # Inherit from Python base class
        pass # Implementation is in the .pyx file

*   **Placeholders:** Replace ``<module>``, ``<name>``, and ``Py<name>`` (Python class name, e.g., ``PyMyEvent``).
*   **``extern from``:** Points Cython to your C++ header.
*   **``cppclass``:** Declares the C++ class and its public members accessible from Python.
*   **``cdef class``:** Declares the Python wrapper, inheriting from ``EventTemplate``.

5. Cython Implementation (``events/<module>/<name>.pyx``)
--------------------------------------------------------

Implement the Python wrapper class.

.. code-block:: python

    # distutils: language=c++
    # cython: language_level=3

    # Import the C++ class definition from the .pxd file
    from AnalysisG.events.<module>.<name> cimport <name>
    # Import the Python base class
    from AnalysisG.core.event_template cimport EventTemplate

    # Define the Python wrapper class
    cdef class Py<name>(EventTemplate):

        # --- Constructor & Destructor ---
        def __cinit__(self):
            # Create an instance of the C++ class
            self.ptr = new <name>() # self.ptr holds the C++ object pointer

        def __init__(self):
            pass # Standard Python init

        def __dealloc__(self):
            # Delete the C++ object when Python object is garbage collected
            del self.ptr

        # --- Properties for Accessing C++ Members ---
        @property
        def EventWeight(self):
            # Expose C++ member 'event_weight' as Python property 'EventWeight'
            return (<<name>*>self.ptr).event_weight

        # Add properties for other exposed members...

        # --- Custom Python Methods (Optional) ---
        # def my_method(self, arg):
        #     # Access C++ members/methods via self.ptr
        #     pass

*   **Placeholders:** Replace ``<module>``, ``<name>``, and ``Py<name>``.
*   **``__cinit__``:** Creates the C++ object (``new <name>()``).
*   **``__dealloc__``:** Deletes the C++ object (``del self.ptr``).
*   **Properties:** Use ``@property`` to provide Pythonic access to C++ members declared in the ``.pxd``. The cast ``(<name>*)self.ptr`` is often needed to access members of the derived C++ class.

6. CMake Configuration (``CMakeLists.txt``)
-------------------------------------------

Add rules to build the C++ library and the Cython extension.

.. code-block:: cmake

    # --------------- DEFINE THE EVENT: <name> ------------------ #
    # Define source files
    set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/<module>/<name>.h)
    set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cxx/<name>.cxx)

    # Create a static library for the C++ code
    add_library(c<name> STATIC ${SOURCE_FILES})

    # Specify include directories for the C++ library
    target_include_directories(c<name> PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Link the event library against the core event template library
    # Add other dependencies (like particle libraries) if needed
    target_link_libraries(c<name> PUBLIC cevent) # Link against base event library

    # Set compiler options (Position Independent Code is usually required)
    target_compile_options(c<name> PRIVATE -fPIC)

    # Call the Cython build helper function (provided by AnalysisG)
    # Args: <cython_module_path> <cython_file_base> <cpp_library_target> <extra_link_libs>
    cmake_language(CALL cybuild "events/<module>/<name>" "events/<module>/<name>" c<name> "")

    # Optional: Install __init__.py/.pxd for the module
    # file(INSTALL ${CMAKE_CURRENT_SOURCE_DIR}/events/<module>/__init__.pxd DESTINATION .)
    # file(INSTALL ${CMAKE_CURRENT_SOURCE_DIR}/events/<module>/__init__.py  DESTINATION .)
    # ------------------------------------------------------------------- #

*   **Placeholders:** Replace ``<module>`` and ``<name>`` consistently.
*   **``add_library``:** Defines the C++ static library (conventionally ``c<name>``).
*   **``target_include_directories``:** Specifies header locations.
*   **``target_link_libraries``:** Links against dependencies (at least ``cevent``). Add links to particle libraries (e.g., ``cMyParticleLib``) if your event uses custom particles.
*   **``cybuild``:** The AnalysisG CMake function that compiles the Cython code and links it against the C++ library (``c<name>``).

After adding these files and configuring CMake, rebuild your AnalysisG project. You should then be able to import and use your new Python event class (``Py<name>``) within the framework.
add_library(c<name> STATIC ${SOURCE_FILES})

# Specify include directories for the C++ library
target_include_directories(c<name> PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link the event library against the core event template library
# Add other dependencies (like particle libraries) if needed
target_link_libraries(c<name> PUBLIC cevent) # Link against base event library

# Set compiler options (Position Independent Code is usually required)
target_compile_options(c<name> PRIVATE -fPIC)

# Call the Cython build helper function (provided by AnalysisG)
# Args: <cython_module_path> <cython_file_base> <cpp_library_target> <extra_link_libs>
cmake_language(CALL cybuild "events/<module>/<name>" "events/<module>/<name>" c<name> "")

# Optional: Install __init__.py/.pxd for the module
# file(INSTALL ${CMAKE_CURRENT_SOURCE_DIR}/events/<module>/__init__.pxd DESTINATION .)
# file(INSTALL ${CMAKE_CURRENT_SOURCE_DIR}/events/<module>/__init__.py  DESTINATION .)
# ------------------------------------------------------------------- #
```

*   **Placeholders:** Replace `<module>` and `<name>` consistently.
*   **`add_library`:** Defines the C++ static library (conventionally `c<name>`).
*   **`target_include_directories`:** Specifies header locations.
*   **`target_link_libraries`:** Links against dependencies (at least `cevent`). Add links to particle libraries (e.g., `cMyParticleLib`) if your event uses custom particles.
*   **`cybuild`:** The AnalysisG CMake function that compiles the Cython code and links it against the C++ library (`c<name>`).

After adding these files and configuring CMake, rebuild your AnalysisG project. You should then be able to import and use your new Python event class (`Py<name>`) within the framework.

Creating Custom C++ Selections for AnalysisG
===========================================

This tutorial outlines the steps to create a custom C++ selection class for the AnalysisG framework, wrap it using Cython, and integrate it using CMake. Selections allow users to define specific criteria for event processing and analysis.

Prerequisites
-------------

*   AnalysisG framework installed.
*   Basic understanding of C++, Cython, and CMake.

Placeholders Used
-----------------

*   `<selection-name>`: The base name for your C++ class and files (e.g., `MySelection`).
*   `<selection-name-python>`: The name for your Python wrapper class (e.g., `MySelectionPython`).
*   `<event-name>`: The specific event class you are working with (e.g., `event_nominal`).
*   `<var-name>`: A placeholder for a custom member variable (e.g., `selected_jets_pt`).
*   `<dependencies>`: Additional CMake link dependencies if needed.

Step 1: Define the C++ Header File (`<selection-name>.h`)
---------------------------------------------------------

Create a header file defining your custom selection class. It must inherit from `selection_template` and override its virtual methods.

```cpp
#ifndef <selection-name>_H
#define <selection-name>_H

// Include the specific event header and the base selection template
#include <<event-name>/event.h>
#include <templates/selection_template.h>

// Include necessary standard library headers
#include <vector>

// Define the custom selection class inheriting from selection_template
class <selection-name> : public selection_template
{
public:
    // Constructor: Set the selection name
    <selection-name>();

    // Destructor: Override is required
    ~<selection-name>() override;

    // clone(): Returns a new instance of this selection (mandatory)
    selection_template* clone() override;

    // selection(): Applies selection criteria to an event (mandatory)
    // Returns true if the event passes the selection, false otherwise.
    bool selection(event_template* ev) override;

    // strategy(): Defines the core logic/analysis strategy for the event (mandatory)
    // Returns true on success, false on failure.
    bool strategy(event_template* ev) override;

    // merge(): Merges data from another instance of this selection (mandatory)
    // Used for combining results, e.g., in parallel processing.
    void merge(selection_template* sl) override;

    // --- Custom Member Variables ---
    // Example: A vector to store some floating-point data
    std::vector<float> <var-name>;
};

#endif // <selection-name>_H
```

Step 2: Implement the C++ Source File (`<selection-name>.cxx`)
------------------------------------------------------------

Implement the methods defined in the header file.

```cpp
#include "<selection-name>.h"

// Constructor implementation: Initialize base class name
<selection-name>::<selection-name>() {
    this->name = "<selection-name>"; // Set the unique name for this selection
}

// Destructor implementation (can be empty if no custom cleanup needed)
<selection-name>::~<selection-name>() {}

// clone() implementation: Return a pointer to a new instance
selection_template* <selection-name>::clone() {
    return (selection_template*)new <selection-name>();
}

// merge() implementation: Combine data from another selection instance
void <selection-name>::merge(selection_template* sl) {
    // Cast the base pointer to the derived type
    <selection-name>* slt = (<selection-name>*)sl;

    // --- Merge Custom Variables ---
    // Example: Use the provided merge_data helper for vectors
    // The third argument is the name used for ROOT output (optional)
    this->merge_data(&this-><var-name>, &slt-><var-name>);
    this->write(&this-><var-name>, "CustomVariableOutputName"); // Register for output
}

// selection() implementation: Define event selection logic
bool <selection-name>::selection(event_template* ev) {
    // Cast the generic event pointer to the specific event type
    <event-name>* evn = (<event-name>*)ev;

    // --- Add Selection Logic Here ---
    // Example: Always pass the event
    bool pass_selection = true;
    // if (evn -> some_property < threshold) { pass_selection = false; }

    return pass_selection;
}

// strategy() implementation: Define the analysis strategy
bool <selection-name>::strategy(event_template* ev) {
    // Cast the generic event pointer to the specific event type
    <event-name>* evn = (<event-name>*)ev;

    // --- Add Analysis Strategy Here ---
    // Example: Populate the custom variable
    // this-><var-name>.push_back(evn -> some_value);

    return true; // Return true if strategy execution was successful
}
```

Step 3: Create the Cython Definition File (`<selection-name>.pxd`)
-----------------------------------------------------------------

This file declares the C++ class structure to Cython, enabling the creation of a Python wrapper.

```python
# distutils: language=c++
# cython: language_level=3

# Import necessary C++ types and the base Cython class
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport selection_template, SelectionTemplate

# Declare the C++ class interface to Cython
cdef extern from "<selection-name>.h":
    cdef cppclass <selection-name>(selection_template):
        # Declare the constructor (add 'except +' for exception handling)
        <selection-name>() except +
        # Declare any public member variables you need to access from Cython/Python
        vector[float] <var-name>

# Declare the Cython wrapper class inheriting from the base SelectionTemplate
cdef class <selection-name-python>(SelectionTemplate):
    # Pointer to the C++ object instance
    cdef <selection-name>* tt
    # Declare methods/properties to expose to Python if needed
    # property custom_var:
    #     def __get__(self):
    #         # Example: return a Python list copy of the C++ vector
    #         return list(self.tt.<var-name>)
```

Step 4: Create the Cython Implementation File (`<selection-name>.pyx`)
-------------------------------------------------------------------

This file implements the Python wrapper class defined in the `.pxd` file.

```python
# distutils: language=c++
# cython: language_level=3

# Import helper functions and the base Cython class
from AnalysisG.core.tools cimport as_dict, as_list # Example helpers
from AnalysisG.core.selection_template cimport SelectionTemplate
# Import the Cython definition
from .<selection-name> cimport <selection-name>

# Implement the Cython wrapper class
cdef class <selection-name-python>(SelectionTemplate):

    # __cinit__: Called upon object creation in Python
    def __cinit__(self):
        # Create a new C++ object instance and assign it to the base pointer 'ptr'
        self.ptr = new <selection-name>()
        # Cast the base pointer to the derived type for specific access
        self.tt = <<selection-name>*>self.ptr

    # __dealloc__: Called when the Python object is garbage collected
    def __dealloc__(self):
        # Delete the C++ object to prevent memory leaks
        # The 'ptr' is managed by the base class, only delete 'tt' if it's
        # different or if custom cleanup is needed beyond deleting 'ptr'.
        # In this typical setup, deleting self.ptr in the base class is sufficient.
        # If self.tt points to the same memory as self.ptr, avoid double deletion.
        # Consider `del self.tt` only if tt manages separate resources.
        # A safer approach might be to rely solely on the base class destructor for self.ptr.
        pass # Rely on base class SelectionTemplate to delete self.ptr

    # Optional: Implement methods to expose C++ functionality or data
    # def get_custom_data(self):
    #     # Example: Return the custom variable as a Python list
    #     return list(self.tt.<var-name>)

    # transform_dict_keys: Override if your C++ class uses std::map
    # with keys that need conversion for Python dictionary compatibility.
    cdef void transform_dict_keys(self):
        # Convert C++ map keys (e.g., std::string) to Python strings if needed
        pass
```

Step 5: Update CMakeLists.txt
-----------------------------

Modify the `CMakeLists.txt` file in the directory containing your selection code (or a parent directory) to build the C++ code and the Cython wrapper.

```cmake
# Define variables for header and source files
set(HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/<selection-name>.h)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/<selection-name>.cxx)

# Add a static library target for the C++ code
add_library(c<selection-name> STATIC ${SOURCE_FILES})

# Specify include directories for the C++ library
target_include_directories(c<selection-name> PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR} # Directory containing <selection-name>.h
    $<INSTALL_INTERFACE:include> # Standard install include path
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> # AnalysisG include path
)

# Link the C++ library against required dependencies
target_link_libraries(c<selection-name> PUBLIC
    cselection_template # Link against the base selection template library
    c<event-name>       # Link against the event library being used
    <dependencies>      # Add any other necessary libraries (e.g., ROOT::Core)
)

# Set compile options (e.g., Position Independent Code for shared libraries)
target_compile_options(c<selection-name> PRIVATE -fPIC)

# Use the AnalysisG cybuild function to create the Cython extension module
# Arguments:
# 1. Target Name: Unique name for the CMake target (e.g., AnalysisG::Selections::<selection-name>)
# 2. Python Module Path: Where the module will be importable from (e.g., AnalysisG.selections.<selection-name>)
# 3. Cython File Base Name: Name of the .pyx/.pxd files (without extension)
# 4. C++ Library Target: The C++ library to link against (created above)
# 5. Python Dependencies: List of other Python modules this depends on (space-separated string)
cmake_language(CALL cybuild
    TARGET AnalysisG::Selections::<selection-name>
    MODULE AnalysisG.selections.<selection-name>
    NAME <selection-name>
    CPP_TARGET c<selection-name>
    PYDEPS "" # Add Python dependencies here if needed
)

# Optional: Install the header file
install(FILES ${HEADER_FILES} DESTINATION include/AnalysisG/selections)
```

Step 6: Build and Use
---------------------

1.  Run CMake from your build directory to configure the project.
2.  Build the project (e.g., using `make` or `ninja`).
3.  In your Python analysis script, import and use the selection:

```python
from AnalysisG.selections.<selection-name> import <selection-name-python>
from AnalysisG.events.<event-name> import <event-name> # Import your event class

# Instantiate the selection
my_selection = <selection-name-python>()

# Instantiate the event object (assuming it's loaded elsewhere)
event = <event-name>()
# ... load event data ...

# Apply the selection and strategy
if my_selection.selection(event):
    my_selection.strategy(event)

# Access results if needed (requires implementing accessors in Cython)
# custom_data = my_selection.get_custom_data()
```

This provides a template for creating custom C++ selections within the AnalysisG framework, bridging the performance of C++ with the flexibility of Python through Cython and CMake. Remember to replace the placeholders with your specific names and logic.




The SelectionTemplate Source Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

C++ Example Interface 
^^^^^^^^^^^^^^^^^^^^^

.. code:: C++ 

    #ifndef example_selection_H
    #define example_selection_H
    
    #include <<event-name>/event.h>
    #include <templates/selection_template.h>
    
    class example_selection: public selection_template
    {
        public:
            example_selection();
            ~example_selection() override; 
            selection_template* clone() override; 
    
            bool selection(event_template* ev) override; 
            bool strategy(event_template* ev) override;
            void merge(selection_template* sl) override;
    
    
            std::vector<float> <var-name>; 
    };

    #endif


.. code:: C++

    #include "<selection-name>.h"

    example_selection::example_selection(){this -> name = "example_selection";}
    example_selection::~example_selection(){}
    
    selection_template* example_selection::clone(){
        return (selection_template*)new example_selection();
    }
    
    void example_selection::merge(selection_template* sl){
        example_selection* slt = (example_selection*)sl; 
    
        // example variable
        this -> merge_data(&this -> <var-name>, &slt -> <var-name>); 
        this -> write(&this -> <var-name>, "some-name-for-ROOT"); 
    }
    
    bool example_selection::selection(event_template* ev){return true;}
    
    bool example_selection::strategy(event_template* ev){
        <event-name>* evn = (<event-name>*)ev; 
    
        return true; 
    }
    
   

Interfacing C++ code with Cython
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: Python

   # distuils: language=c++
   # cython: language_level=3
   # example_selection.pxd
   
   from libcpp.map cimport map
   from libcpp.vector cimport vector
   from libcpp.string cimport string
   from AnalysisG.core.selection_template cimport *
   
   cdef extern from "example_selection.h":
       cdef cppclass example_selection(selection_template):
           example_selection() except +
   
   cdef class ExampleSelection(SelectionTemplate):
       cdef example_selection* tt



.. code:: Python

   # distutils: language=c++
   # cython: language_level=3
   # example_selection.pyx
   
   from AnalysisG.core.tools cimport as_dict, as_list
   from AnalysisG.core.selection_template cimport *
   
   cdef class ExampleSelection(SelectionTemplate):
       def __cinit__(self):
           self.ptr = new example_selection()
           self.tt = <example_selection*>self.ptr
   
       def __dealloc__(self): del self.tt
   
       cdef void transform_dict_keys(self):
           #convert map keys to python string
           pass
   

 

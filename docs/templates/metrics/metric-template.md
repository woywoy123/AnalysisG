Creating Custom C++ Metrics for AnalysisG
=========================================

This tutorial guides you through creating a custom metric in C++ for the AnalysisG framework, wrapping it with Cython for Python integration, and building it using CMake.

Why C++ Metrics?
----------------

While AnalysisG allows metrics written purely in Python, implementing computationally intensive metrics in C++ can offer significant performance advantages, especially when processing large datasets.

Prerequisites
-------------

*   A working AnalysisG development environment.
*   Basic understanding of C++, Cython, and CMake.
*   Familiarity with the AnalysisG `metric_template` concept.

Let's assume you want to create a metric named `MyEfficiency`.

Step 1: Define the C++ Header (`include/metrics/MyEfficiency.h`)
-----------------------------------------------------------------

Create a header file for your C++ metric class. This class must inherit from `metric_template`.

.. code-block:: cpp
    :caption: include/metrics/MyEfficiency.h

    #ifndef MYEFFICIENCY_METRIC_H
    #define MYEFFICIENCY_METRIC_H

    #include <templates/metric_template.h> // Base class header
    #include <string>
    #include <vector> // If needed

    // Declare your C++ metric class
    class MyEfficiency_metric : public metric_template
    {
    public:
         // Constructor
         MyEfficiency_metric();
         // Virtual destructor (important!)
         ~MyEfficiency_metric() override;
         // Clone method for copying
         MyEfficiency_metric* clone() override;

         // --- Required methods from metric_template ---
         void define_variables() override; // Register output variables
         void event() override;            // Process event data
         void batch() override;            // Process batch data (optional)
         void end() override;              // Final calculations
         void define_metric(metric_t* mtx) override; // Access event/batch data

    private:
         // --- Member Variables ---
         std::string mode = "";
         float event_efficiency = 0.0f;
         float total_passed = 0.0f;
         int total_events = 0;
         // Add other necessary variables
    };

    #endif // MYEFFICIENCY_METRIC_H

Step 2: Implement the C++ Logic (`src/metrics/MyEfficiency.cxx`)
-----------------------------------------------------------------

Implement the methods declared in the header file.

.. code-block:: cpp
    :caption: src/metrics/MyEfficiency.cxx

    #include <metrics/MyEfficiency.h>
    #include <vector>
    #include <numeric> // For example calculations

    // Constructor: Set the metric name and initialize variables
    MyEfficiency_metric::MyEfficiency_metric() {
         this->name = "MyEfficiency"; // Unique metric name
         this->event_efficiency = 0.0f;
         this->total_passed = 0.0f;
         this->total_events = 0;
    }

    // Destructor: Clean up resources if needed
    MyEfficiency_metric::~MyEfficiency_metric() {}

    // Clone: Return a new instance
    MyEfficiency_metric* MyEfficiency_metric::clone() {
         return new MyEfficiency_metric();
    }

    // Define Variables: Register outputs for the ROOT file
    void MyEfficiency_metric::define_variables() {
         // register_output("branch_name", "type_code", &member_variable)
         this->register_output("event_eff", "F", &this->event_efficiency);
         this->register_output("total_passed", "F", &this->total_passed);
         // Register more variables as needed
    }

    // Define Metric: Access data from the current event/batch
    void MyEfficiency_metric::define_metric(metric_t* mtx) {
         this->mode = mtx->mode(); // Get current mode ("training", "validation", etc.)

         // Example: Retrieve data needed for calculations
         // bool passed_selection = mtx->get<bool>(event_enum::passed_selection, "CutX");
         // Store data in member variables if needed across methods
    }

    // Event: Calculate metrics for the current event
    void MyEfficiency_metric::event() {
         // --- Perform event-level calculations ---
         // Example: Replace with actual efficiency calculation logic
         bool condition_met = true; // Placeholder
         this->event_efficiency = condition_met ? 1.0f : 0.0f;

         // --- Update global aggregates ---
         if (condition_met) {
              this->total_passed += 1.0f;
         }
         this->total_events++;

         // --- Write event-level results to ROOT file ---
         this->write("event_eff", "F", &this->event_efficiency);
    }

    // Batch: Optional batch-level processing
    void MyEfficiency_metric::batch() {
         // Implement if needed
    }

    // End: Perform final calculations and write summary results
    void MyEfficiency_metric::end() {
         // --- Final calculations ---
         float overall_efficiency = (this->total_events > 0) ? (this->total_passed / this->total_events) : 0.0f;

         // --- Write global/summary results ---
         // Often uses accumulate=true for histograms/graphs or writes final values
         this->write("total_passed", "F", &this->total_passed);
         // You might register and write the overall efficiency too:
         // this->register_output("overall_eff", "F", &this->overall_efficiency_member);
         // this->overall_efficiency_member = overall_efficiency;
         // this->write("overall_eff", "F", &this->overall_efficiency_member);
    }

Step 3: Create the Cython Wrapper (`path/to/your/MyEfficiencyMetric.pyx`)
--------------------------------------------------------------------------

This file bridges the C++ code to Python using Cython.

.. code-block:: cython
    :caption: path/to/your/MyEfficiencyMetric.pyx

    # distutils: language=c++
    # cython: language_level=3

    from AnalysisG.core.metric_template cimport MetricTemplate
    from AnalysisG.core.tools cimport metric_t # If needed directly
    from libcpp.string cimport string

    # Import the C++ class definition
    cdef extern from "<metrics/MyEfficiency.h>":
         cdef cppclass MyEfficiency_metric(metric_template):
              MyEfficiency_metric() except+ # Declare constructor

    # Define the Python wrapper class inheriting from AnalysisG's base
    cdef class MyEfficiencyMetric(MetricTemplate):
         def __cinit__(self):
              # Create an instance of the C++ class when the Python object is created
              self.ptr = new MyEfficiency_metric()
              # Store the base pointer (as expected by MetricTemplate)
              self.mtr = <metric_template*>self.ptr

         # Add any Python-specific methods or properties if needed
         # (Often not necessary if all logic is in C++)

Step 4: Configure the Build with CMake (`CMakeLists.txt`)
----------------------------------------------------------

Add instructions to your `CMakeLists.txt` to build the C++ library and the Cython extension.

.. code-block:: cmake
    :caption: CMakeLists.txt fragment

    # --- Build the C++ Static Library ---

    # Define the C++ source file(s)
    set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/metrics/MyEfficiency.cxx)

    # Create a static library for the metric
    add_library(cmetric_MyEfficiency STATIC ${SOURCE_FILES})

    # Specify include directories
    target_include_directories(cmetric_MyEfficiency
         PRIVATE include/metrics # Location of MyEfficiency.h
         PUBLIC include          # Location of metric_template.h, etc.
    )

    # Link against the base metric template library
    target_link_libraries(cmetric_MyEfficiency PUBLIC cmetric_template)

    # Set compiler options (optional)
    target_compile_options(cmetric_MyEfficiency PRIVATE -fPIC -Wall -Wextra -pedantic)

    # --- Build the Cython Extension Module ---
    # This uses a hypothetical 'cybuild' function. Adapt to your project's setup.
    # Common tools include scikit-build or custom CMake functions.

    # Example using a custom function 'cybuild'
    # Arguments might be: PyxSource, OutputDir, ModuleName, CppDependency
    cmake_language(
         CALL cybuild
         "path/to/your/MyEfficiencyMetric.pyx" # Input Cython file
         "AnalysisG/metrics"                   # Output directory (relative to site-packages)
         "metric_MyEfficiency"                 # Name of the Python module (.so file)
         "cmetric_MyEfficiency"                # C++ library to link against
    )

    # --- Ensure Installation (Example) ---
    # You might also need rules to install the generated .so file
    # install(FILES ${CMAKE_BINARY_DIR}/path/to/metric_MyEfficiency.so DESTINATION ...)


.. note::
    The `cybuild` command is illustrative. Your project might use different CMake functions or tools (like `scikit-build`) to handle Cython compilation and linking. Consult your project's build system documentation.

Step 5: Using the Metric in AnalysisG
-------------------------------------

After successfully building and installing your project, you can import and use the metric in your AnalysisG Python scripts just like any other metric:

.. code-block:: python

    from AnalysisG.core import Analysis
    from AnalysisG.metrics import MyEfficiencyMetric # Import your new metric

    # Instantiate the analysis object
    ana = Analysis()

    # Add your custom metric instance
    eff_metric = MyEfficiencyMetric()
    ana.AddMetric(eff_metric)

    # Configure the rest of your analysis...
    # ana.InputSample(...)
    # ana.EventStop = 100
    # ana.Launch()

Your C++ metric (`MyEfficiency_metric`) will now be executed as part of the AnalysisG event loop, leveraging C++ performance while being controlled from Python.

Tools Module
===========

The Tools module provides general utility functions and helper classes for various tasks across the AnalysisG framework.

Overview
--------

This module implements a collection of utility functions, algorithms, and helper classes that are used throughout the framework. It offers functionality that doesn't fit neatly into other specialized modules but is essential for the overall operation of the framework.

Key Components
-------------

tools class
~~~~~~~~~

.. doxygenclass:: tools
   :members:
   :protected-members:
   :undoc-members:

Main Functionalities
-------------------

Data Processing Utilities
~~~~~~~~~~~~~~~~~~~~~

The module provides utility functions for data processing:

- Data normalization and standardization
- Missing value handling
- Data format conversion
- Vector manipulation operations

File and Path Management
~~~~~~~~~~~~~~~~~~~~~

Utilities for managing files and directory paths:

- ``ensure_directory()``: Creates directories if they don't exist
- Path manipulation and normalization
- File existence checking and validation
- File type detection

String Manipulation
~~~~~~~~~~~~~~~

String handling utilities commonly used in the framework:

- String tokenization and parsing
- Regular expression pattern matching
- String formatting and conversion
- Case manipulation (upper, lower, title case)

Mathematical Operations
~~~~~~~~~~~~~~~~~~

Common mathematical functions beyond standard library offerings:

- Statistical functions (mean, median, percentiles)
- Special mathematical functions relevant for physics
- Numerical stability enhancements
- Random number generation with physics distributions

Logging and Reporting
~~~~~~~~~~~~~~~~~

Tools for logging and generating reports:

- Log message formatting
- Log level control
- Performance timing utilities
- Progress reporting

Usage Example
------------

.. code-block:: cpp

    #include <tools/tools.h>
    
    void process_data_example() {
        tools* utility = new tools();
        
        // Ensure output directory exists
        std::string output_dir = "./results/analysis_123";
        utility->ensure_directory(output_dir);
        
        // Generate a unique identifier
        std::string unique_id = utility->generate_uuid();
        std::cout << "Generated unique ID: " << unique_id << std::endl;
        
        // Format a timestamp
        std::string timestamp = utility->timestamp();
        std::cout << "Current time: " << timestamp << std::endl;
        
        // Normalize a vector of values
        std::vector<float> data = {1.2, 3.5, 2.1, 5.7, 4.3};
        std::vector<float> normalized = utility->normalize(data);
        
        // Format a JSON string
        std::map<std::string, std::string> info;
        info["model"] = "GNN";
        info["dataset"] = "top_tagging";
        info["accuracy"] = "0.923";
        std::string json_str = utility->to_json(info);
        
        delete utility;
    }
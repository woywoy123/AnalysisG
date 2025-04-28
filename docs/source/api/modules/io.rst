IO Module
=========

The IO module manages reading and writing data in various formats commonly used in high energy physics, such as ROOT and HDF5.

Overview
--------

This module provides a unified interface for data input and output operations, abstracting away the complexities of different file formats. It offers seamless conversion between various data storage formats and the internal data structures used within AnalysisG.

Key Components
-------------

io class
~~~~~~~

.. doxygenclass:: io
   :members:
   :protected-members:
   :undoc-members:

Main Functionalities
-------------------

File Handling
~~~~~~~~~~~

The module provides methods for opening, reading, writing, and closing files:

- ``start()``: Opens a file for reading or writing
- ``end()``: Closes an open file and releases resources
- Support for both ROOT and HDF5 file formats

Data Reading
~~~~~~~~~~

Methods for reading data from files into memory:

- ``read()``: Reads data from a specified dataset or branch
- Support for various data types, including vectors and custom structures
- Automatic type conversion between file storage and internal representation

Data Writing
~~~~~~~~~~

Methods for writing data from memory to files:

- ``write()``: Writes data to a specified dataset or branch
- Support for structured data organization in output files
- Efficient serialization of complex data structures

ROOT File Specific Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

Specialized functionality for working with ROOT files:

- ``root_begin()``: Initializes ROOT file processing
- ``scan_keys()``: Discovers available branches and leaves in ROOT files
- ``root_key_paths()``: Generates path mapping for ROOT objects

Dataset Management
~~~~~~~~~~~~~~~

Functions for working with datasets within a file:

- ``dataset_names()``: Lists available datasets in a file
- ``dataset()``: Creates or accesses a specific dataset

Usage Example
------------

.. code-block:: cpp

    #include <io/io.h>
    
    // Reading data from a ROOT file
    void read_example() {
        io* reader = new io();
        
        // Open a ROOT file for reading
        reader->start("data.root", "READ");
        
        // Initialize ROOT file processing
        reader->root_begin();
        
        // Scan available keys/branches
        reader->scan_keys();
        
        // Read data into a vector
        std::vector<float> pt_values;
        reader->read(&pt_values, "particle_pt");
        
        // Close the file
        reader->end();
        
        delete reader;
    }
    
    // Writing data to an HDF5 file
    void write_example() {
        io* writer = new io();
        
        // Open an HDF5 file for writing
        writer->start("output.h5", "WRITE");
        
        // Create and write a dataset
        std::vector<double> energies = {10.5, 20.3, 15.7, 30.2};
        writer->write(&energies, "particle_energies");
        
        // Close the file
        writer->end();
        
        delete writer;
    }
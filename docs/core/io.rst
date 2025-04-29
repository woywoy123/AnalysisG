I/O Module
=========

The I/O module handles all input and output operations for data files in the AnalysisG framework.

Overview
--------

The io class provides a comprehensive interface for reading from ROOT files, writing to HDF5 format, and handling various data formats used in physics analysis.

Key Features
-----------

* **ROOT File Reading**: Access physics data from ROOT files with selective field reading
* **HDF5 Support**: Export and import data structures to the efficient HDF5 format
* **Metadata Extraction**: Extract metadata from physics samples
* **Tree Navigation**: Navigate ROOT file trees and access branches and leaves
* **Caching**: Cache processed data for improved performance

Core Methods
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - ``start()``
     - Initialize a file for reading or writing
   * - ``end()``
     - Close file and finalize operations
   * - ``read()``
     - Read data from a file into the provided structure
   * - ``write()``
     - Write data from a structure to a file
   * - ``check_root_file_paths()``
     - Validate and process provided ROOT file paths
   * - ``cache()``
     - Cache data for future access

Implementation Details
--------------------

.. code-block:: cpp

   class io: public notification, public tools {
   public:
       io();
       ~io();
       
       void start(std::string filename, std::string mode);
       void end();
       
       template <typename T>
       bool read(T* dst, std::string name);
       
       template <typename T>
       bool write(T* src, std::string name);
       
       void check_root_file_paths();
       
       // Configuration and state
       std::vector<std::string> trees;
       std::vector<std::string> branches;
       std::vector<std::string> leaves;
       std::map<std::string, bool> root_files;
   };

Example Usage
------------

.. code-block:: cpp

   // Create I/O object
   io* reader = new io();
   
   // Configure trees and leaves to read
   reader->trees = {"nominal"};
   reader->leaves = {"pt", "eta", "phi", "energy"};
   
   // Add ROOT files
   reader->root_files["/path/to/file.root"] = true;
   
   // Validate paths
   reader->check_root_file_paths();
   
   // Read events
   reader->start();
   // Process data...
   reader->end();

HDF5 Operations
--------------

The I/O module can efficiently store and retrieve graph data using HDF5:

.. code-block:: cpp

   // Writing graph data to HDF5
   io* writer = new io();
   writer->start("graphs.h5", "write");
   writer->write(&graph_data, "graphs");
   writer->end();
   
   // Reading graph data from HDF5
   io* reader = new io();
   reader->start("graphs.h5", "read");
   reader->read(&graph_data, "graphs");
   reader->end();

ROOT File Support
---------------

The I/O module handles ROOT files with special consideration for:

* Event-based data structures
* Physics metadata
* Cross-section information
* Weight normalization
* Detector simulation information
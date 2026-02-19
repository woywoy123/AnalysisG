I/O Module (C++)
=================

The I/O module provides ROOT file input/output operations.

Overview
--------

Located in ``src/AnalysisG/modules/io/``, this module implements C++ I/O functionality 
for reading and writing ROOT files:

- TTree reading and writing
- Branch access and manipulation
- Efficient data extraction
- Memory-mapped file access

Purpose
-------

The I/O module provides low-level ROOT file operations:

- Direct TTree access
- Branch iteration
- Type-safe data extraction
- Optimized file reading

Implementation Files
--------------------

**C++ Implementation**

- ``src/AnalysisG/modules/io/cxx/*.cxx`` - I/O implementations
- ``src/AnalysisG/modules/io/include/io/*.h`` - I/O headers

Key Classes
-----------

**ROOT Reader**

Reads data from ROOT files:

.. code-block:: cpp

   class root_reader {
   public:
       // Open file
       void open(std::string filename);
       
       // Get tree
       TTree* get_tree(std::string tree_name);
       
       // Read branch
       template<typename T>
       T* read_branch(std::string branch_name);
       
       // Iterate entries
       size_t get_entries();
       void load_entry(size_t index);
   };

**ROOT Writer**

Writes data to ROOT files:

.. code-block:: cpp

   class root_writer {
   public:
       // Create file
       void create(std::string filename);
       
       // Create tree
       TTree* create_tree(std::string tree_name);
       
       // Add branch
       template<typename T>
       void add_branch(std::string branch_name, T* data);
       
       // Fill tree
       void fill();
       
       // Save and close
       void close();
   };

Features
--------

**Efficient Reading**

- Selective branch reading
- Memory-mapped access
- Batch reading
- Caching strategies

**Type Safety**

- Template-based type inference
- Compile-time type checking
- Runtime type validation
- Automatic conversions

**Memory Management**

- Automatic buffer allocation
- Smart pointer usage
- Resource cleanup
- Memory pool optimization

Usage Example
-------------

**Reading ROOT Files**

.. code-block:: cpp

   #include <io/root_reader.h>
   
   // Create reader
   root_reader reader;
   reader.open("data.root");
   
   // Get tree
   TTree* tree = reader.get_tree("nominal");
   
   // Read branches
   auto pt = reader.read_branch<std::vector<float>>("jets_pt");
   auto eta = reader.read_branch<std::vector<float>>("jets_eta");
   
   // Iterate entries
   for (size_t i = 0; i < reader.get_entries(); ++i) {
       reader.load_entry(i);
       // Access data
       for (size_t j = 0; j < pt->size(); ++j) {
           double jet_pt = (*pt)[j];
           double jet_eta = (*eta)[j];
       }
   }

**Writing ROOT Files**

.. code-block:: cpp

   #include <io/root_writer.h>
   
   // Create writer
   root_writer writer;
   writer.create("output.root");
   
   // Create tree
   TTree* tree = writer.create_tree("results");
   
   // Add branches
   float met, ht;
   writer.add_branch("met", &met);
   writer.add_branch("ht", &ht);
   
   // Fill tree
   for (auto& event : events) {
       met = event.met;
       ht = event.ht;
       writer.fill();
   }
   
   // Save
   writer.close();

Integration with Cython
-----------------------

The C++ I/O module is wrapped in Python via ``src/AnalysisG/core/io.pyx``:

.. code-block:: python

   from AnalysisG.core.io import IO
   
   # Create IO instance
   io = IO(["data.root"])
   io.Trees = ["nominal"]
   io.Leaves = ["jets_pt", "jets_eta"]
   
   # Iterate events
   for event in io:
       pt = event[b"nominal.jets_pt.jets_pt"]

Performance Optimization
------------------------

**Caching**

- Branch data caching
- Entry prefetching
- Basket cache tuning
- Memory limits

**Parallel Reading**

- Multi-threaded file access
- Concurrent tree reading
- Thread-safe buffer management

**Compression**

- Automatic decompression
- Compression level configuration
- Format detection

ROOT Integration
----------------

The module integrates with ROOT features:

- TChain for multiple files
- Friends for associated trees
- TTreeFormula for expressions
- TSelector for processing

Error Handling
--------------

Robust error handling for:

- File not found
- Corrupted files
- Missing trees/branches
- Type mismatches
- Memory errors

See Also
--------

* :doc:`../core/io` - Python IO wrapper
* :doc:`event` - Event template using I/O
* :doc:`meta` - Metadata handling

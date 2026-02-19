Sample Tracer Module (C++)
===========================

The Sample Tracer module provides sample tracking and provenance information.

Overview
--------

Located in ``src/AnalysisG/modules/sampletracer/``, this module implements sample 
tracking functionality in C++:

- Sample identification
- File provenance tracking
- Event counting per sample
- Weight accumulation

Purpose
-------

The sampletracer module enables:

- Tracking which samples events come from
- Maintaining sample statistics
- Debugging data flow
- Cutflow tables per sample

Implementation Files
--------------------

**C++ Implementation**

- ``src/AnalysisG/modules/sampletracer/cxx/*.cxx`` - Sample tracer implementations
- ``src/AnalysisG/modules/sampletracer/include/generators/*.h`` - Tracer headers

Key Classes
-----------

**sample_tracer**

Sample tracking:

.. code-block:: cpp

   class sample_tracer {
   public:
       // Sample identification
       std::string sample_name;
       std::vector<std::string> files;
       
       // Event counting
       long total_events;
       long processed_events;
       std::map<std::string, long> events_per_file;
       
       // Weight tracking
       double sum_weights;
       double sum_weights_squared;
       std::map<std::string, double> weights_per_file;
       
       // Methods
       void add_file(std::string filename);
       void record_event(std::string filename, double weight);
       void print_summary();
   };

Usage Example
-------------

**Tracking Samples**

.. code-block:: cpp

   #include <generators/sample_tracer.h>
   
   // Create tracer
   sample_tracer tracer;
   tracer.sample_name = "ttbar";
   
   // Add files
   tracer.add_file("file1.root");
   tracer.add_file("file2.root");
   
   // Record events
   for (auto& event : events) {
       tracer.record_event(event.filename, event.weight);
   }
   
   // Print summary
   tracer.print_summary();

Sample Statistics
-----------------

Track detailed statistics:

.. code-block:: cpp

   struct SampleStats {
       long total_events;
       long passed_events;
       double efficiency;
       double sum_weights;
       double avg_weight;
       double weight_variance;
   };
   
   SampleStats stats = tracer.get_statistics();

Cutflow Per Sample
------------------

Track cutflow for each sample:

.. code-block:: cpp

   class SampleCutflow {
   public:
       std::string sample_name;
       std::map<std::string, long> cut_counts;
       std::map<std::string, double> cut_weights;
       
       void record_cut(std::string cut_name, double weight) {
           cut_counts[cut_name]++;
           cut_weights[cut_name] += weight;
       }
       
       void print_cutflow() {
           for (auto& [cut, count] : cut_counts) {
               std::cout << cut << ": " << count 
                        << " events, weight: " << cut_weights[cut] 
                        << std::endl;
           }
       }
   };

File Provenance
---------------

Track file-level information:

.. code-block:: cpp

   struct FileInfo {
       std::string filename;
       long events_in_file;
       long events_processed;
       double file_weight;
       std::time_t process_time;
   };
   
   std::vector<FileInfo> provenance = tracer.get_provenance();

Integration with Analysis
-------------------------

Sample tracer integrates with analysis:

.. code-block:: cpp

   // In analysis
   analysis.add_sample_tracer("signal", signal_files);
   analysis.add_sample_tracer("background", background_files);
   
   // After processing
   auto signal_stats = analysis.get_sample_stats("signal");
   auto bkg_stats = analysis.get_sample_stats("background");

Multi-Sample Comparison
-----------------------

Compare multiple samples:

.. code-block:: cpp

   std::map<std::string, sample_tracer> tracers;
   
   for (auto& [name, files] : samples) {
       sample_tracer tracer;
       tracer.sample_name = name;
       for (auto& file : files) {
           tracer.add_file(file);
       }
       tracers[name] = tracer;
   }
   
   // Print comparison
   print_sample_comparison(tracers);

Output:

.. code-block:: text

   Sample      Events    Passed    Efficiency    Sum Weights
   --------------------------------------------------------
   ttbar       1000000   450000    45.0%         450000.0
   Wjets       5000000   1200000   24.0%         1200000.0
   QCD         10000000  500000    5.0%          500000.0

See Also
--------

* :doc:`analysis` - Analysis using sample tracing
* :doc:`meta` - Metadata for samples

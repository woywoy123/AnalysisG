Notification Module (C++)
==========================

The Notification module provides progress reporting and status updates.

Overview
--------

Located in ``src/AnalysisG/modules/notification/``, this module implements notification 
functionality in C++:

- Progress reporting
- Status updates
- Error notifications
- Logging capabilities

Purpose
-------

The notification module enables:

- Real-time progress tracking
- Status messages during analysis
- Error and warning reporting
- Logging to files and console

Implementation Files
--------------------

**C++ Implementation**

- ``src/AnalysisG/modules/notification/cxx/*.cxx`` - Notification implementations
- ``src/AnalysisG/modules/notification/include/notification/*.h`` - Notification headers

**Python Binding**

- ``src/AnalysisG/core/notification.pyx`` - Cython wrapper
- ``src/AnalysisG/core/notification.pxd`` - Cython declarations

Key Classes
-----------

**notification**

Notification handler:

.. code-block:: cpp

   class notification {
   public:
       // Progress reporting
       void progress(std::string message, double percent);
       void status(std::string message);
       
       // Logging
       void info(std::string message);
       void warning(std::string message);
       void error(std::string message);
       void debug(std::string message);
       
       // Configuration
       void set_log_level(int level);  // 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
       void set_log_file(std::string path);
       void enable_console(bool enable);
   };

Usage Example
-------------

**Progress Reporting**

.. code-block:: cpp

   #include <notification/notification.h>
   
   notification notif;
   
   // Report progress
   for (size_t i = 0; i < total; ++i) {
       // Process item
       double percent = 100.0 * i / total;
       notif.progress("Processing events", percent);
   }

**Status Updates**

.. code-block:: cpp

   notif.status("Starting analysis...");
   notif.status("Loading data...");
   notif.status("Training model...");
   notif.status("Analysis complete!");

**Logging**

.. code-block:: cpp

   notif.info("Analysis started");
   notif.warning("Low statistics in bin 5");
   notif.error("Failed to load file");
   notif.debug("Variable value: " + std::to_string(x));

Log Levels
----------

Different log levels for controlling verbosity:

- **DEBUG** (0): Detailed debugging information
- **INFO** (1): General informational messages
- **WARNING** (2): Warning messages
- **ERROR** (3): Error messages

.. code-block:: cpp

   // Set log level
   notif.set_log_level(1);  // INFO and above

Output Destinations
-------------------

**Console Output**

.. code-block:: cpp

   // Enable/disable console
   notif.enable_console(true);

**File Logging**

.. code-block:: cpp

   // Log to file
   notif.set_log_file("analysis.log");

**Both**

.. code-block:: cpp

   // Log to both console and file
   notif.enable_console(true);
   notif.set_log_file("analysis.log");

Message Formatting
------------------

Messages are formatted with:

- Timestamp
- Log level
- Source location (optional)
- Message content

Example output:

.. code-block:: text

   [2024-02-19 15:30:45] INFO: Analysis started
   [2024-02-19 15:30:46] WARNING: Low statistics in bin 5
   [2024-02-19 15:30:50] ERROR: Failed to load file

Progress Bar
------------

Sophisticated progress reporting:

.. code-block:: cpp

   notification notif;
   notif.start_progress("Processing", total_items);
   
   for (size_t i = 0; i < total_items; ++i) {
       // Process item
       notif.update_progress(i + 1);
   }
   
   notif.end_progress();

Output:

.. code-block:: text

   Processing: [████████████████████          ] 65% (650/1000)

Multi-threaded Support
----------------------

Thread-safe notification:

.. code-block:: cpp

   // Thread-safe logging
   #pragma omp parallel for
   for (size_t i = 0; i < n; ++i) {
       notif.info("Thread " + std::to_string(omp_get_thread_num()) + 
                  " processing item " + std::to_string(i));
   }

Custom Handlers
---------------

Register custom notification handlers:

.. code-block:: cpp

   class CustomHandler {
   public:
       void handle(std::string message, int level) {
           // Custom handling (e.g., send to monitoring system)
       }
   };
   
   notif.register_handler(new CustomHandler());

Integration with Analysis
-------------------------

Notifications integrate with analysis workflow:

.. code-block:: cpp

   // In analysis loop
   analysis.notification.info("Starting sample: " + sample_name);
   analysis.notification.progress("Processing events", percent_complete);
   analysis.notification.warning("Low weight: " + std::to_string(weight));

Timed Operations
----------------

Time operations and report:

.. code-block:: cpp

   notif.start_timer("data_loading");
   load_data();
   double elapsed = notif.stop_timer("data_loading");
   notif.info("Data loading took " + std::to_string(elapsed) + " seconds");

Conditional Logging
-------------------

Log only when conditions are met:

.. code-block:: cpp

   // Log every N iterations
   if (i % 100 == 0) {
       notif.info("Processed " + std::to_string(i) + " events");
   }
   
   // Log on errors only
   if (status != SUCCESS) {
       notif.error("Operation failed with status " + std::to_string(status));
   }

Integration with Python
-----------------------

The C++ notification is wrapped in Python:

.. code-block:: python

   from AnalysisG.core.notification import Notification
   
   notif = Notification()
   notif.info("Starting analysis")
   notif.progress("Processing", 50.0)

See Also
--------

* :doc:`../core/notification` - Python Notification wrapper
* :doc:`analysis` - Analysis using notifications

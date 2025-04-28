Analysis Module
==============

The Analysis module serves as the orchestration layer of the AnalysisG framework, coordinating the execution of data processing, model training, and evaluation workflows.

Overview
--------

This module provides the central ``analysis`` class that connects all other components of the framework. It manages the execution flow, from loading data to training models and producing final analysis results.

Key Components
-------------

analysis class
~~~~~~~~~~~~~

.. doxygenclass:: analysis
   :members:
   :protected-members:
   :private-members:
   :undoc-members:

Main Functionalities
-------------------

Workflow Management
~~~~~~~~~~~~~~~~~

The Analysis module coordinates the entire analysis workflow by:

- Loading and preprocessing data
- Building graph representations
- Configuring and training machine learning models
- Executing event selections
- Computing and recording metrics
- Generating output visualizations

Model Session Management
~~~~~~~~~~~~~~~~~~~~~~

The module provides methods to build, train, and evaluate model sessions:

- ``build_model_session()``: Initialize model training session
- ``execution()``: Execute model inference on data

Progress Tracking
~~~~~~~~~~~~~~~

Several methods monitor and report the progress of analysis tasks:

- ``progress()``: Reports numerical progress metrics
- ``progress_mode()``: Reports the current mode of operation
- ``progress_report()``: Generates detailed progress reports
- ``is_complete()``: Checks completion status of analysis tasks

Usage Example
------------

.. code-block:: cpp

    #include <AnalysisG/analysis.h>

    int main() {
        analysis* an = new analysis();
        
        // Configure analysis settings
        settings_t settings;
        settings.output_path = "./results";
        settings.run_name = "top_tagging";
        settings.epochs = 20;
        an->import_settings(&settings);
        
        // Execute the analysis workflow
        an->run();
        
        delete an;
        return 0;
    }
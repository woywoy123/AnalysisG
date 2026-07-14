Minimal Working Example
=======================

The ``AnalysisG`` framework is designed so that complex, multi-threaded C++ MapReduce routines and ROOT IO handlers can be executed using simple Python boilerplate. 

To understand exactly how the C++ backend operates under the hood, we will analyze a real-world minimal working example derived from the ``studies/topreconstruction/main.py`` script, and compare it side-by-side with the C++ handlers it invokes in ``src/AnalysisG/modules/``.

The Python Driver
-----------------

Here is the core logic required to run a complete GNN event mapping and selection pipeline across a dataset of ROOT files.

.. code-block:: python

    from AnalysisG import Analysis
    from AnalysisG.events.gnn import EventGNN
    from AnalysisG.selections.performance.topefficiency.topefficiency import TopEfficiency

    def main():
        # Initialize the C++ driver
        ana = Analysis()
        ana.Threads = 4

        # Register event schemas and ROOT files
        ev = EventGNN()
        dataset_path = "/home/tnom6927/scratch/*.root"
        ana.AddEvent(ev, dataset_path)
        ana.AddSamples(dataset_path, "MyTopDataset")

        # Attach custom event selection logic
        sel = TopEfficiency()
        ana.AddSelection(sel)
        
        # Configure IO to persist aggregated C++ data
        ana.SaveSelectionToROOT = True

        # Trigger C++ MapReduce
        ana.Start()

    if __name__ == "__main__":
        main()


Step-by-Step C++ Backend Translation
------------------------------------

When the Python ``main()`` script executes, it rapidly delegates memory management and thread orchestration down to the C++ backend.

### 1. The ``Analysis`` Initialization
When ``ana = Analysis()`` is instantiated, it spawns a native C++ ``analysis`` wrapper (found in ``src/AnalysisG/modules/analysis/cxx/analysis.cxx``). 
By setting ``ana.Threads = 4``, the Python attribute setter directly modifies the C++ ``settings_t`` struct inside the wrapper, preparing the internal ``container_t`` thread pool size.

### 2. Event Registration (``AddEvent`` and ``AddSamples``)
When ``ana.AddEvent(ev, dataset_path)`` is called, the C++ driver utilizes the ``SampleTracer`` module (``src/AnalysisG/modules/sampletracer/cxx/sampletracer.cxx``).
- **File Hashing**: The C++ code globs the ROOT files at ``dataset_path``, opens them momentarily, and computes a unique internal hash for each file.
- **Event Binding**: The ``EventGNN`` Cython object is passed down as a pointer. The C++ framework registers this object as the "decoder ring" for translating ROOT ``TTree`` branches into mapped C++ structs in memory.

### 3. Selection Attachment
Calling ``ana.AddSelection(sel)`` passes the ``TopEfficiency`` module into the C++ pipeline. 
Internally, the driver stores a pointer to the ``selection_template`` base class. This is where the MapReduce framework comes alive. The C++ backend maps the user's ``TopEfficiency::selection()`` method to be the mapper (executed concurrently), and ``TopEfficiency::merge()`` to be the reducer (executed synchronously on the main thread).

### 4. Triggering the Pipeline (``Start()``)
Calling ``ana.Start()`` initiates the most intense C++ routines.

**The Map Phase**
Inside the ``container`` module, the C++ framework spawns 4 worker ``std::thread`` instances. The ``dataloader`` module reads chunks of the ROOT file (using ROOT's optimized TTreeReader if available), deserializes the data using the ``EventGNN`` template, and feeds each reconstructed event object to a waiting thread. The thread executes ``sel->selection(event)``.

**The Reduce Phase**
Once all events in a chunk or file are processed, the threads join. The main C++ thread then executes ``sel->merge(thread_local_selection)``. This is where the ``mergecast`` C++ module heavily utilizes templates to aggregate the thread-local ``std::map`` and ``std::vector`` results safely.

**The IO Phase**
Finally, because ``ana.SaveSelectionToROOT = True`` was enabled, the driver delegates to the ``writer.cxx`` module. The writer scrapes the aggregated data out of the Selection object and generates a new `.root` file containing the processed metrics, completing the pipeline.

.. _exports_root:

ROOT Interactions and I/O
=========================

Integration with the CERN ROOT framework is managed by the ``io`` class (primarily within ``src/AnalysisG/modules/io/cxx/root.cxx``). While often referred to conceptually as part of the I/O system's "export/writer" routines, this module's primary directive is highly optimized reading, tree traversal, metadata scraping, and memory caching of input ROOT files.

Dynamic PCM Generation
----------------------
ROOT's object serialization requires C++ dictionaries to parse advanced custom objects effectively. The ``io::trigger_pcm()`` method ensures these objects can be read smoothly:

- It invokes ``TSystem`` to dynamically set up ACLiC (Automatic Compiler of Libraries for CINT/Cling) configuration.
- It triggers parallel background compilations (``.pcm`` and ``.so`` dictionaries) for necessary objects like ``meta_t`` and ``weights_t``, loading them instantly without stalling the run loop.

Tree and Branch Extraction
--------------------------
To maintain agnosticism and avoid rigid hardcoded loops, ``AnalysisG`` dynamically walks the ROOT ``TFile`` hierarchies:

1. **Key Scanning**: The ``io::scan_keys()`` function explores ``TDirectory`` and ``TTree`` content dynamically. It validates structures recursively and constructs internal maps detailing paths to every accessible tree and branch.
2. **Metadata Scraping**: As the hierarchies are traversed, specific objects (such as `AnalysisTracking`, `EventLoop_FileExecuted`, or `MetaData`) are intercepted. A robust ``meta`` scraping engine aggregates statistics like sum-of-weights for seamless luminosity and cross-section normalizations.
3. **TLeaf Iteration Setup**: During ``io::root_begin()``, pointers to requested branches (``TBranch``) and their fundamental leaves (``TLeaf``) are instantiated. The backend prepares ``data_t`` caching structs for highly performant read cycles using continuous sequential memory. 

Downstream Integration
----------------------
Rather than writing output structures back into ``.root`` flat trees, the ``io`` class translates incoming ROOT event fragments directly into native PyTorch tensors and C++ ``graph_t`` nodes. This guarantees downstream operations remain universally portable and ready for HDF5 caching.

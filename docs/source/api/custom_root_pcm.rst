.. _custom-root-pcm:

ROOT to PCM Mapping
===================

When interfacing C++ with ROOT, compiling custom classes into dictionaries is necessary for ROOT's I/O to recognize and serialize them. 
AnalysisG handles this automatically through its internal PCM (Pre-Compiled Module) mapping and generation infrastructure.

The ``trigger_pcm()`` Mechanism
-------------------------------
The core logic resides in ``io::trigger_pcm()`` (found in ``src/AnalysisG/modules/io/cxx/root.cxx``). 
This routine automates dictionary generation using ROOT's ``TSystem`` interface:

1. **Path setup**: The build directory is configured using ``gSystem->SetBuildDir()``, specifying a ``pcm/`` subdirectory to hold the generated dictionary and module files.
2. **ACLiC configuration**: The dynamic path is set, and ``gSystem->SetAclicMode(TSystem::kOpt)`` is invoked to optimize the compilation process.
3. **Threaded Compilation**: 
   Dictionaries for various custom structures (such as ``meta_t`` and ``weights_t`` from ``meta.h``) are generated concurrently using ``std::thread``. 
   The ``buildDict`` and ``buildAll`` routines invoke ROOT's dictionary generator (``rootcling`` or ``ACLiC`` equivalents) behind the scenes.
4. **Integration**: By pre-building these module files and dynamically linking them, ROOT can safely parse, serialize, and deserialize C++ structures like ``std::map``, ``std::vector``, and AnalysisG structs directly.

Why is this necessary?
----------------------
ROOT files (``.root``) containing custom objects need their layout mapped out via dictionary files. By triggering the PCM build dynamically, AnalysisG allows users to introduce new C++ data types into their TTrees without having to manually run ``rootcling``.

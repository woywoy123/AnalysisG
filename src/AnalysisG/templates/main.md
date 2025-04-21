Introduction to EventTemplates and ParticleTemplates
====================================================

Given that C++ is a compiled language, the compiler needs to know about the source files which hold the definitions of the objects.
To achieve this, the package uses `CMake` to link everything together and make the package available for the compiler.

The Source Files 
----------------

But first there are some files that need to be created for the event and particles to be defined.

- Create a new folder within ```console src/AnalysisG/events/```.
- Within the folder create the following files:

  - `CMakeLists.txt` (copy the one from `bsm_4tops` for example)
  - `event.cxx`
  - `event.h`
  - `particles.cxx`
  - `particles.h`

- Outside of the event folder, create the following files:

  - `event_<some name>.pxd`
  - `event_<some name>.pyx`

- Modify the `CMakeLists.txt` within the events folder and add the event to the list.

TLDR (Too Long Do Read)
-----------------------

For a quick introduction by example, checkout the template code under `src/AnalysisG/templates`.
A brief explanation:

- **events**: Templates used to define an event implementation.
- **graphs**: Templates used to define graphs for GNN training and inference.
- **selections**: Templates used for in-depth studies using the associated event implementation.
- **model**: Templates for implementing a custom GNN model.

Container Module
================

The ``container`` class manages per-file typed data containers used for
storing events, graphs, and selections during compilation.  One
``container`` instance is created per ROOT/HDF5 file by ``sampletracer``.

``entry_t`` — Per-hash slot
----------------------------

``entry_t`` is the fundamental storage record inside a ``container``.
Each unique event hash maps to one ``entry_t`` that holds the associated
event, graph, and selection template pointers.

.. doxygenstruct:: entry_t
   :project: AnalysisG
   :members:

``container`` — Per-file object store
--------------------------------------

.. doxygenclass:: container
   :project: AnalysisG
   :members:
   :protected-members:
   :undoc-members:

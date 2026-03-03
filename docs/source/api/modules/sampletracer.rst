Sample Tracer Module
====================

The ``sampletracer`` class tracks ROOT/HDF5 sample metadata, manages the
event–graph–selection population pipeline, and coordinates multi-threaded
file compilation.  It owns one ``container`` per input file and is the
primary in-memory store for the entire analysis job.

.. doxygenclass:: sampletracer
   :project: AnalysisG
   :members:
   :protected-members:
   :undoc-members:

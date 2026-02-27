Graph Operations
================

Batched graph-level operations (edge/node single-counting aggregation,
etc.) implemented as CUDA/C++ kernels in the ``graph_`` internal
namespace and exposed via ``pyc::graph``.

.. doxygennamespace:: graph_
   :project: AnalysisG
   :members:
   :undoc-members:

Doxygen Source
--------------

The documentation above is derived from the following ``.dox`` annotation file(s):

``graph.dox``

.. literalinclude:: ../../../../src/AnalysisG/pyc/graph/include/graph/graph.dox
   :language: c
   :caption:


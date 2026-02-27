pyc Interface
=============

The ``pyc`` namespace is the public C++/CUDA interface for tensor-based
particle physics computations.  It contains sub-namespaces ``transform``,
``physics``, ``operators``, ``nusol``, and ``graph``.

The ``neutrino`` class is a lightweight ``particle_template`` subclass
for reconstructed neutrinos.

.. doxygenclass:: neutrino
   :project: AnalysisG
   :members:
   :undoc-members:

.. doxygennamespace:: pyc
   :project: AnalysisG
   :members:
   :undoc-members:

Doxygen Source
--------------

The documentation above is derived from the following ``.dox`` annotation file(s):

``pyc.dox``

.. literalinclude:: ../../../../src/AnalysisG/pyc/interface/include/pyc/pyc.dox
   :language: c
   :caption:


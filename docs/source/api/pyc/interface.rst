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

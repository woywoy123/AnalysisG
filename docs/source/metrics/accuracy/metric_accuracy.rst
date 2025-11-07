metric_accuracy.pyx
===================

**File Path**: ``src/AnalysisG/metrics/accuracy/metric_accuracy.pyx``

**File Type**: Cython Source

**Lines**: 152

Description
-----------

from AnalysisG.core.roc cimport *
cdef tuple mx_index(vector[double]* sc):
cdef collector* cl = vl.cl

**Cython Imports**:

- ``*``

**Python Imports**:

- ``*``
- ``AnalysisG.core.roc``

Classes
-------

``AccuracyMetric``
~~~~~~~~~~~~~~~~~~

Class defined in this file.

Functions/Methods
-----------------

- ``Postprocessing()``
- ``__cinit__()``


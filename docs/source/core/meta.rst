meta.pyx
========

**File Path**: ``src/AnalysisG/core/meta.pyx``

**File Type**: Cython Source

**Lines**: 811

Description
-----------

from AnalysisG.core.tools cimport *
from AnalysisG.core.meta cimport *
from AnalysisG.core.notification cimport *

**Cython Imports**:

- ``*``
- ``dereference``
- ``meta_t,``
- ``pair,``
- ``prange``
- ``string``
- ``vector``

**Python Imports**:

- ``*``
- ``AnalysisG``
- ``AnalysisG.core.meta``
- ``AnalysisG.core.notification``
- ``AnalysisG.core.structs``
- ``AnalysisG.core.tools``
- ``PyAMI:``
- ``auth_pyami``
- ``cython.operator``
- ``cython.parallel``
- ``dereference``
- ``h5py``
- ``http.client``
- ``libcpp.map``
- ``libcpp.string``
- ``libcpp.vector``
- ``meta_t,``
- ``pair,``
- ``pickle``
- ``prange``

Classes
-------

``Data``
~~~~~~~~

Class defined in this file.

``Meta``
~~~~~~~~

Class defined in this file.

``MetaLookup``
~~~~~~~~~~~~~~

Class defined in this file.

``ami_client``
~~~~~~~~~~~~~~

Class defined in this file.

``atlas``
~~~~~~~~~

Class defined in this file.

``httpx``
~~~~~~~~~

Class defined in this file.

Functions/Methods
-----------------

- ``AtlasRelease()``
- ``CrossSection()``
- ``DatasetName()``
- ``ExpectedEvents()``
- ``FetchMeta()``
- ``Files()``
- ``GenerateData()``
- ``GetSumOfWeights()``
- ``MetaCachePath()``
- ``PDF()``
- ``SumOfWeights()``
- ``__add__()``
- ``__call__()``
- ``__cinit__()``
- ``__dealloc__()``
- ``__init__()``
- ``__meta__()``
- ``__radd__()``
- ``__reduce__()``
- ``__str__()``
- ``amiStatus()``
- ``amitag()``
- ``beamType()``
- ``beam_energy()``
- ``campaign()``
- ``completion()``
- ``conditionsTag()``
- ``config()``
- ``connect()``
- ``crossSection()``


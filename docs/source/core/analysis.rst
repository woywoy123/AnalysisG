analysis.pyx
============

**File Path**: ``src/AnalysisG/core/analysis.pyx``

**File Type**: Cython Source

**Lines**: 348

Description
-----------

from AnalysisG.core.meta cimport *
from AnalysisG.core.tools cimport *
from AnalysisG.core.lossfx cimport *
from AnalysisG.core.event_template cimport *
from AnalysisG.core.graph_template cimport *

**Cython Imports**:

- ``*``
- ``analysis``
- ``bool``
- ``pair,``
- ``string``

**Python Imports**:

- ``*``
- ``AnalysisG.core.analysis``
- ``AnalysisG.core.event_template``
- ``AnalysisG.core.graph_template``
- ``AnalysisG.core.lossfx``
- ``AnalysisG.core.meta``
- ``AnalysisG.core.metric_template``
- ``AnalysisG.core.model_template``
- ``AnalysisG.core.selection_template``
- ``AnalysisG.core.tools``
- ``analysis``
- ``bool``
- ``gc``
- ``libcpp``
- ``libcpp.map``
- ``libcpp.string``
- ``pair,``
- ``pickle``
- ``sleep``
- ``string``

Classes
-------

``Analysis``
~~~~~~~~~~~~

Class defined in this file.

Functions/Methods
-----------------

- ``AddEvent()``
- ``AddGraph()``
- ``AddMetric()``
- ``AddModel()``
- ``AddModelInference()``
- ``AddSamples()``
- ``AddSelection()``
- ``BatchSize()``
- ``BuildCache()``
- ``ContinueTraining()``
- ``DebugMode()``
- ``Epochs()``
- ``Evaluation()``
- ``FetchMeta()``
- ``GetMetaData()``
- ``GraphCache()``
- ``MaxRange()``
- ``NumExamples()``
- ``OutputPath()``
- ``PreTagEvents()``
- ``SaveSelectionToROOT()``
- ``SetLogY()``
- ``Start()``
- ``SumOfWeightsTreeName()``
- ``Targets()``
- ``Threads()``
- ``TrainSize()``
- ``Training()``
- ``TrainingDataset()``
- ``Validation()``


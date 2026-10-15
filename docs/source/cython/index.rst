Cython API and inheritance
==========================

AnalysisG Cython code is implemented primarily in ``src/AnalysisG/**/*.pyx``
with declarations in ``.pxd`` files. These wrappers expose C++ classes and
template families to Python.

Core wrapper bases
------------------

- ``EventTemplate`` (``src/AnalysisG/core/event_template.pyx``)
- ``ParticleTemplate`` (``src/AnalysisG/core/particle_template.pyx``)
- ``GraphTemplate`` (``src/AnalysisG/core/graph_template.pyx``)
- ``SelectionTemplate`` (``src/AnalysisG/core/selection_template.pyx``)
- ``ModelTemplate`` (``src/AnalysisG/core/model_template.pyx``)
- ``MetricTemplate`` (``src/AnalysisG/core/metric_template.pyx``)

Derived wrapper classes
-----------------------

Event wrappers (``EventTemplate``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``BSM4Tops`` (``events/bsm_4tops/event_bsm_4tops.pyx``)
- ``ExpMC20`` (``events/exp_mc20/event_exp_mc20.pyx``)
- ``EventGNN`` (``events/gnn/event_gnn.pyx``)
- ``SSML_MC20`` (``events/ssml_mc20/event_ssml_mc20.pyx``)

Particle wrappers (``ParticleTemplate``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``Top``, ``Child``, ``PhysicsDetector``, ``PhysicsTruth``, ``Electron``,
  ``Muon``, ``Jet`` (``events/exp_mc20/particle_exp_mc20.pyx``)
- ``Top``, ``ZPrime`` (``events/gnn/particle_gnn.pyx``)
- ``Neutrino``, ``Particle`` in neutrino selection modules
  (``selections/neutrino/*``)

Graph wrappers (``GraphTemplate``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``GraphTops``, ``GraphChildren``, ``GraphTruthJets``, ``GraphTruthJetsNoNu``,
  ``GraphJets``, ``GraphJetsNoNu``, ``GraphDetectorLep``, ``GraphDetector``
  (``graphs/bsm_4tops/graph_bsm_4tops.pyx``)
- ``GraphJets``, ``GraphJetsNoNu``, ``GraphDetectorLep``, ``GraphDetector``
  in ``graphs/exp_mc20`` and ``graphs/ssml_mc20``.

Selection wrappers (``SelectionTemplate``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Implemented across ``src/AnalysisG/selections`` including:

- ``Regions``, ``MET``, ``ChildrenKinematics``, ``DecayModes``, ``MissingET``,
  ``Parton``, ``TopJets``, ``TopKinematics``, ``TopMatching``,
  ``TopTruthJets``, ``ZPrime``, ``NuNuCombinatorial``, ``MultiSol``,
  ``Validation``, ``TopEfficiency``.

Model wrappers (``ModelTemplate``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``RecursiveGraphNeuralNetwork``
- ``RecyclX``
- ``Grift``

Metric wrappers (``MetricTemplate``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``AccuracyMetric``
- ``PageRankMetric``
- ``TopEfficiencyMetric``

Plotting inheritance
^^^^^^^^^^^^^^^^^^^^

- ``TH1F(BasePlotting)``, ``TH2F(BasePlotting)``, ``TLine(BasePlotting)``
- ``ROC(TLine)``

Cython-to-C++ binding pattern
-----------------------------

The Cython wrappers bind C++ classes through ``.pxd`` declarations, for example:

- ``ExpMC20(EventTemplate)`` wraps ``cppclass exp_mc20(event_template)``
  (``events/exp_mc20/event_exp_mc20.pxd``).
- ``ParticleTemplate`` wraps ``cppclass particle_template``
  (``core/particle_template.pxd``).
- ``EventTemplate`` wraps ``cppclass event_template``
  (``core/event_template.pxd``).

Limitations
-----------

Because these are ``cdef`` classes backed by compiled extensions, Sphinx cannot
reliably introspect runtime signatures without building the extension modules.
This documentation therefore uses source-level mapping from ``.pyx``/``.pxd``
files, preserving the repository’s current class relationships.

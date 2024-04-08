MC-16 Simulation Truth Matching
===============================

This section of the studies is dedicated to understanding whether the truth matching to tops is consistent.
Studies are detailed below and with their respective figures being listed for each study.
To reproduce these studies, samples need to be generated using an AnalysisTop derived codebase, which is linked below:

https://github.com/woywoy123/BSM4tops-GNN-ntuples/tree/main

.. toctree::
   :titlesonly:

   mc16/z-prime/main.rst
   mc16/top-kinematics/main.rst
   mc16/top-matching/main.rst
   mc16/children-kinematics/main.rst
   mc16/decay-modes/main.rst
   mc16/truth-event/main.rst
   mc16/truth-jets/main.rst
   mc16/jets/main.rst
   mc16/other/main.rst

MC-20 Simulation Truth Matching
===============================

This section of the studies involves the study of how generator tops are matched to detector level physics objects.
Studies are described below with their respective figure names.
To reproduce these studies, special samples need to be produced with another framework which, can be found under:

https://gitlab.cern.ch/atlas-phys/exot/hqt/ana-exot-2022-44/common-framework/-/tree/add-top-matching?ref_type=heads

.. toctree::
   :titleonly:

   mc20/top/main.rst

Glossary
--------

- **Truth Tops**: Tops-Quarks within the sample which have already emitted gluons (Final State Radiation).
- **Truth Top Children (Truth Children)**: Particles originating from the Truth Tops, if any of the particles are a W-Boson, then its children are used.
- **Truth Jets**: Partons after they undergo hadronization without any detector convolutions. 
- **Jets**: Partons after they undergo hadronization with detector convolutions.
- **Signal/Resonance Tops**: Top-Quarks produced from the Z/H Beyond Standard Model Particle.
- **Spectator Tops**: Tops-Quarks which are the by-product of producing a Z/H boson.



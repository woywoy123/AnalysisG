Same-Sign Multi-Lepton MC20 Event (``ssml_mc20``)
==================================================

The ``ssml_mc20`` package provides the event and particle classes for the
same-sign multi-lepton MC20 analysis.  Import with::

    from AnalysisG.events.ssml_mc20 import SSML_MC20

SSML_MC20
---------

``SSML_MC20`` is an :class:`~AnalysisG.core.event_template.EventTemplate`
subclass wrapping ``<ssml_mc20/event.h>``.  It inherits all base event
properties and registers its particle collections inside ``CompileEvent``.
No additional scalar fields are defined beyond the base class.

Particle Classes
----------------

electron
^^^^^^^^

Reconstructed electron for the SSML_MC20 event.  Inherits all
:class:`~AnalysisG.core.particle_template.ParticleTemplate` kinematics;
no additional fields are defined beyond the base class.

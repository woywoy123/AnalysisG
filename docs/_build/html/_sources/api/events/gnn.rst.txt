GNN Inference Event (``gnn``)
==============================

The ``gnn`` package provides the event and particle classes used during
GNN inference runs.  Import with::

    from AnalysisG.events.gnn import EventGNN

EventGNN
--------

``EventGNN`` is an :class:`~AnalysisG.core.event_template.EventTemplate`
subclass wrapping ``<inference/gnn-event.h>``.  It inherits all base
event properties (``index``, ``weight``, ``Tree``, ``Trees``, etc.) and
exposes no additional scalar fields — particle collections are registered
via :meth:`~AnalysisG.core.event_template.EventTemplate.register_particle`
inside ``CompileEvent``.

Particle Classes
----------------

Top
^^^

Inferred top quark.  Inherits all
:class:`~AnalysisG.core.particle_template.ParticleTemplate` kinematics;
no additional fields.

ZPrime
^^^^^^

Inferred Z′ resonance particle.  Inherits all
:class:`~AnalysisG.core.particle_template.ParticleTemplate` kinematics;
no additional fields.

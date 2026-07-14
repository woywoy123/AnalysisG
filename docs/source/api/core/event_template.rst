EventTemplate (Python)
======================

The ``EventTemplate`` Cython class wraps the C++ ``event_template``.
User event classes must subclass it, call ``register_particle`` in
``__cinit__``, and override ``CompileEvent`` for post-processing.

Special Methods
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Signature
     - Description
   * - ``__hash__() → int``
     - Integer hash derived from the first 8 hex digits of ``self.hash``.
   * - ``__eq__(other) → bool``
     - Equality based on C++ ``operator==``.
   * - ``is_self(inpt) → bool``
     - Return ``True`` if ``inpt`` is an instance or subclass of
       ``EventTemplate``.

Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Property
     - Type
     - Description
   * - ``index``
     - ``float``
     - Sequential event index within the current ROOT file.
   * - ``weight``
     - ``float``
     - Event weight (default ``0.0``; must be set in ``CompileEvent``).
   * - ``Tree``
     - ``str``
     - Active ROOT TTree name for this event (read/write).
   * - ``Trees``
     - ``list[str]``
     - List of TTree names this event type reads from.  Set in the subclass
       constructor before ``Analysis.Start()``.
   * - ``Branches``
     - ``list[str]``
     - List of TBranch names to read.
   * - ``Name``
     - ``str``
     - Human-readable event class name (read-only, from the C++ ``name``
       field).

C++ Interface (called from subclass ``build``)
-----------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method / field
     - Description
   * - ``register_particle(ptr)``
     - Register a ``std::map<std::string, ParticleType*>*`` so that
       ``build_event`` populates it from ROOT data.
   * - ``CompileEvent()``
     - Post-processing hook called once after all particles are built.
       Override in subclasses to set ``self.weight``, compute derived
       quantities, etc.
   * - ``add_leaf(key, leaf)``
     - Register an additional ROOT leaf under the given key.

Event Template
==============

``event_template`` is the base class for all user-defined event types.  It
manages the mapping between ROOT branch/leaf data (stored in ``element_t``
structs) and the C++ particle collections registered by subclasses, and exposes
kinematic metadata via ``cproperty`` accessors.

Class: ``event_template``
--------------------------

**Header:** ``<templates/event_template.h>``

**Inheritance:** ``tools``

Properties
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 25 57

   * - Property
     - Type
     - Description
   * - ``trees``
     - ``std::vector<std::string>``
     - List of ROOT tree names this event reads from.  Writable.
   * - ``branches``
     - ``std::vector<std::string>``
     - List of ROOT branch-path filters.  Writable.
   * - ``leaves``
     - ``std::vector<std::string>``
     - Derived list of fully-resolved ROOT leaf paths (read-only; computed from registered particles).
   * - ``name``
     - ``std::string``
     - Human-readable event-class name (used as HDF5 dataset label).  Writable.
   * - ``hash``
     - ``std::string``
     - 18-character hex identifier for this event.  Settable and gettable.
   * - ``tree``
     - ``std::string``
     - The ROOT tree name under which this event was read.  Gettable/Settable.
   * - ``weight``
     - ``double``
     - Monte Carlo event weight.  Default ``0.0``.
   * - ``index``
     - ``long``
     - Sequential event index within the file.  Default ``0``.

Public Fields
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Field
     - Type
     - Description
   * - ``m_trees``
     - ``std::map<std::string, std::string>``
     - Internal map of tree-key → ROOT tree name.
   * - ``m_branches``
     - ``std::map<std::string, std::string>``
     - Internal map of branch-key → ROOT branch name.
   * - ``m_leaves``
     - ``std::map<std::string, std::string>``
     - Resolved leaf-key → ROOT leaf path (populated by ``register_particle``).
   * - ``meta_data``
     - ``meta*``
     - Pointer to the dataset metadata object (set by ``io`` / ``sampletracer``).
   * - ``filename``
     - ``std::string``
     - Path of the ROOT file from which this event was read.
   * - ``data``
     - ``event_t``
     - Raw event-level data struct (index, weight, hash, …).

Public Methods
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Signature
     - Description
   * - ``template<G> void register_particle(std::map<std::string, G*>* object)``
     - Registers a particle collection: creates a prototype ``G``, extracts its
       ``type`` and leaf map, and stores a pointer to *object* for population
       during ``build_event``.
   * - ``template<G> void deregister_particle(std::map<std::string, G*>* object)``
     - Deletes and clears all entries in *object*.  Call in the event destructor.
   * - ``void add_leaf(std::string key, std::string leaf = "")``
     - Adds an event-level ROOT leaf mapping (for event-level scalars, not per-particle).
   * - ``virtual void build(element_t* el)``
     - Called once per event entry.  Override to create/populate particle maps
       from the ROOT branch data in *el*.
   * - ``virtual void CompileEvent()``
     - Called after ``build``.  Override to compute derived quantities (topology,
       secondary particles, truth matching).
   * - ``virtual event_template* clone()``
     - Returns a heap-allocated copy of this event.  Override in subclasses.
       Caller owns the returned pointer.
   * - ``std::map<std::string, event_template*> build_event(std::map<std::string, data_t*>* evnt)``
     - Framework-internal: builds a map of fully populated event objects from
       the raw ``data_t`` map produced by ``io``.
   * - ``std::vector<particle_template*> multi_neutrino(...)``
     - Runs the neutrino combinatorial solver for dilepton events.
       Parameters: *targets* (W-decay products), *phi* (MET φ), *met* (MET pt),
       optional *mt* (top mass, MeV), *mw* (W mass, MeV), *violation*, *limit*.
   * - ``bool operator == (event_template& p)``
     - Hash-equality comparison.
   * - ``void flush_particles()``
     - Frees all registered particle collections (called during cleanup).


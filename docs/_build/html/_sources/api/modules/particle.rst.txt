Particle Template
=================

``particle_template`` is the C++ base for every user-defined particle type.
It stores a ``particle_t`` data struct (4-momentum + flags) and exposes
kinematic properties via the ``cproperty<T,G>`` getter/setter mechanism so that
subclasses can override individual components without changing the interface.

The class also manages topology helpers (``register_parent``, ``register_child``,
``DeltaR``), ROOT-branch leaf registration (``add_leaf``, ``apply_type_prefix``),
and in-place four-momentum addition (``operator +=``, ``iadd``).

Class: ``particle_template``
-----------------------------

**Header:** ``<templates/particle_template.h>``

**Inheritance:** ``tools``

Constructors
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Signature
     - Description
   * - ``particle_template()``
     - Default constructor; initialises ``data`` to NaN kinematics and ``is_add = true``.
   * - ``particle_template(particle_t* p)``
     - Initialises from a serialised ``particle_t`` struct (used during de-serialisation).
   * - ``particle_template(particle_template* p, bool dump = false)``
     - Copy constructor; ``dump=true`` copies without leaf mappings.
   * - ``particle_template(double px, double py, double pz, double e)``
     - Constructs from Cartesian four-momentum (all in MeV).
   * - ``particle_template(double px, double py, double pz)``
     - Constructs massless particle from 3-momentum (``e = ||p||``).

Kinematic Properties (``cproperty``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All kinematic properties are readable and writable.  The framework converts
between cylindrical (pt, η, φ, E) and Cartesian (px, py, pz, E)
representations on demand via ``to_polar()`` / ``to_cartesian()``.

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Property
     - Type
     - Description
   * - ``e``
     - ``double``
     - Energy in MeV.
   * - ``pt``
     - ``double``
     - Transverse momentum ``√(px²+py²)`` in MeV.
   * - ``eta``
     - ``double``
     - Pseudorapidity ``-ln(tan(θ/2))``.
   * - ``phi``
     - ``double``
     - Azimuthal angle ``atan2(py, px)`` in radians.
   * - ``px``
     - ``double``
     - x-component of 3-momentum in MeV.
   * - ``py``
     - ``double``
     - y-component of 3-momentum in MeV.
   * - ``pz``
     - ``double``
     - z-component of 3-momentum (``pt·sinh(η)``) in MeV.
   * - ``mass``
     - ``double``
     - Invariant mass ``√(E²−||p||²)`` in MeV (``nan`` if space-like).
   * - ``P``
     - ``double``
     - 3-momentum magnitude ``√(px²+py²+pz²)`` in MeV.  Read-only.
   * - ``beta``
     - ``double``
     - Relativistic β = ||p||/E.  Read-only.

Meta-Data Properties
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Property
     - Type
     - Description
   * - ``pdgid``
     - ``int``
     - PDG particle identifier.  Default ``0``.
   * - ``symbol``
     - ``std::string``
     - LaTeX symbol, e.g. ``"t"`` or ``"\\nu"``.
   * - ``charge``
     - ``double``
     - Electric charge in units of elementary charge.
   * - ``hash``
     - ``std::string``
     - 18-character hex string uniquely identifying this particle in an event.
       Read-only — computed from kinematic content.
   * - ``type``
     - ``std::string``
     - String type prefix set by ``apply_type_prefix()`` (e.g. ``"top"``).
   * - ``index``
     - ``int``
     - Position index within the parent event particle collection.  Default ``-1``.

Truth / Flag Properties (read-only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All flags are derived from ``particle_t.pdgid``:

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Property
     - Type
     - Description
   * - ``is_b``
     - ``bool``
     - ``true`` if the particle is a b-quark (abs(pdgid) = 5).
   * - ``is_lep``
     - ``bool``
     - ``true`` if the particle is a charged lepton (abs(pdgid) ∈ {11,13,15}).
   * - ``is_nu``
     - ``bool``
     - ``true`` if the particle is a neutrino (abs(pdgid) ∈ {12,14,16}).
   * - ``is_add``
     - ``bool``
     - ``true`` if the particle was added to an event (not matched to truth).
       Default ``true`` for default-constructed particles.
   * - ``lep_decay``
     - ``bool``
     - ``true`` if the top quark decayed leptonically (set by the event builder).

Topology Properties
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Property
     - Type
     - Description
   * - ``parents``
     - ``std::map<std::string, particle_template*>``
     - Map of parent particles by their hash.
   * - ``children``
     - ``std::map<std::string, particle_template*>``
     - Map of child particles by their hash.
   * - ``m_parents``
     - ``std::map<std::string, particle_template*>``
     - Internal backing store for ``parents`` (public for framework use).
   * - ``m_children``
     - ``std::map<std::string, particle_template*>``
     - Internal backing store for ``children``.

Public Methods
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Signature
     - Description
   * - ``double DeltaR(particle_template* p)``
     - Returns ``√(Δη² + Δφ²)`` between ``this`` and *p*.
   * - ``bool operator == (particle_template& p)``
     - Hash-equality comparison.
   * - ``template<g> g operator + (g& p)``
     - Returns a new particle of type *g* whose 4-momentum is the sum of
       ``this`` and *p* (Cartesian addition).
   * - ``void operator += (particle_template* p)``
     - Adds *p*'s 4-momentum to ``this`` (calls ``iadd``).
   * - ``void iadd(particle_template* p)``
     - In-place 4-momentum addition (Cartesian).
   * - ``bool register_parent(particle_template* p)``
     - Adds *p* to ``m_parents`` by hash.  Returns ``false`` if already present.
   * - ``bool register_child(particle_template* p)``
     - Adds *p* to ``m_children`` by hash.  Returns ``false`` if already present.
   * - ``void add_leaf(std::string key, std::string leaf = "")``
     - Registers ROOT branch mapping: ``key`` is the property name;
       ``leaf`` is the ROOT branch suffix (with leading underscore, e.g.
       ``"_pt"``).  The full branch name is resolved by ``apply_type_prefix()``.
   * - ``void apply_type_prefix()``
     - Prepends ``type`` to every leaf suffix registered via ``add_leaf``.
       For example, with ``type="top"`` and leaf ``"_pt"`` the resolved
       branch name becomes ``"top_pt"``.
   * - ``std::map<std::string, std::map<std::string, particle_t>> __reduce__()``
     - Serialises the particle to a nested map suitable for HDF5 storage.
   * - ``virtual void build(std::map<std::string, particle_template*>* event, element_t* el)``
     - Override in subclasses to populate the particle from ROOT branch data
       held in *el* (called by the event builder).
   * - ``virtual particle_template* clone()``
     - Override in subclasses to return a heap-allocated copy of the particle.
       The caller is responsible for ownership.
   * - ``bool is(std::vector<int> p)``
     - Returns ``true`` if ``abs(pdgid)`` is contained in *p*.

Public Data
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Field
     - Type
     - Description
   * - ``data``
     - ``particle_t``
     - Raw kinematic and flag storage struct.
   * - ``leaves``
     - ``std::map<std::string, std::string>``
     - Maps property key → resolved ROOT branch name (populated by ``add_leaf`` + ``apply_type_prefix``).
   * - ``_is_serial``
     - ``bool``
     - Set to ``true`` by the framework after serialisation.
   * - ``_is_marked``
     - ``bool``
     - Set to ``true`` by ``selection_template::sum`` on synthesised combination
       particles; prevents ``safe_delete`` from freeing them twice.


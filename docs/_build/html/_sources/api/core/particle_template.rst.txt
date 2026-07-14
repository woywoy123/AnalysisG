ParticleTemplate (Python)
=========================

The ``ParticleTemplate`` Cython class is the Python-side wrapper around the C++
``particle_template``.  User particle classes must subclass it and override
``Type`` to set the ROOT branch prefix.

Constructor
-----------

``ParticleTemplate(inpt=None)`` — default-constructs the underlying C++ object.
Pass a serialised ``dict`` (from ``__reduce__``) to restore a saved state.

Operators / Special Methods
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Signature
     - Description
   * - ``__add__(other: ParticleTemplate) → ParticleTemplate``
     - Return a new particle whose four-momentum is the sum of ``self`` and
       ``other``.  Children and Parents lists are merged.
   * - ``__iadd__(other: ParticleTemplate) → ParticleTemplate``
     - Add ``other``'s four-momentum into ``self`` in-place and merge its
       Children and Parents.
   * - ``__eq__(other) → bool``
     - Equality based on the C++ ``operator==`` (compares all four-momentum
       components and flags).
   * - ``__hash__() → int``
     - Integer hash derived from the first 8 hex digits of ``self.hash``.
   * - ``is_self(inpt) → bool``
     - Return ``True`` if ``inpt`` is an instance or subclass of
       ``ParticleTemplate``.
   * - ``clone() → ParticleTemplate``
     - Return a new default-constructed instance of the same concrete subclass
       with the same ``Type`` string.
   * - ``DeltaR(other: ParticleTemplate) → float``
     - Compute angular separation :math:`\Delta R = \sqrt{(\Delta\eta)^2 + (\Delta\phi)^2}`.
   * - ``dump(path, name) / load(path, name)``
     - Pickle serialise / restore the particle to/from ``path/name.pkl``.

Kinematic Properties (read/write)
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Property
     - Type
     - Description
   * - ``px``
     - ``float``
     - Cartesian x-momentum in MeV.
   * - ``py``
     - ``float``
     - Cartesian y-momentum in MeV.
   * - ``pz``
     - ``float``
     - Cartesian z-momentum in MeV.
   * - ``e``
     - ``float``
     - Energy in MeV.
   * - ``pt``
     - ``float``
     - Transverse momentum.
   * - ``eta``
     - ``float``
     - Pseudorapidity.
   * - ``phi``
     - ``float``
     - Azimuthal angle in radians.
   * - ``Mass``
     - ``float``
     - Invariant mass in MeV (computed on demand, ``nan`` until set).
   * - ``charge``
     - ``float``
     - Electric charge.
   * - ``pdgid``
     - ``int``
     - PDG particle ID code.
   * - ``symbol``
     - ``str``
     - Particle symbol string (e.g. ``"t"`` for top quark).
   * - ``Type``
     - ``str``
     - ROOT branch type prefix.  Must be set by each concrete subclass.
       Used by ``apply_type_prefix()`` to build full branch names.
   * - ``index``
     - ``int``
     - Sequential particle index within the event (read from ``particle_t.index``).

Flag Properties (read/write)
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Property
     - Type
     - Description
   * - ``is_lep``
     - ``bool``
     - ``True`` if this particle is classified as a lepton.
   * - ``is_nu``
     - ``bool``
     - ``True`` if this particle is classified as a neutrino.
   * - ``is_b``
     - ``bool``
     - ``True`` if this particle is classified as a b-quark/jet.
   * - ``is_add``
     - ``bool``
     - ``True`` if this particle is an additional (non-prompt) particle.
       Default is ``True``.
   * - ``LeptonicDecay``
     - ``bool``
     - ``True`` if this particle decays leptonically.
   * - ``lepdef``
     - ``list[int]``
     - PDG IDs that define "lepton" for the ``is_lep`` test.
   * - ``nudef``
     - ``list[int]``
     - PDG IDs that define "neutrino" for the ``is_nu`` test.

Topology Properties
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Property
     - Type
     - Description
   * - ``Children``
     - ``list[ParticleTemplate]``
     - List of child particles registered via ``register_child``.
   * - ``Parents``
     - ``list[ParticleTemplate]``
     - List of parent particles registered via ``register_parent``.
   * - ``hash``
     - ``str``
     - 18-character unique hex identifier (``"0x"`` + 16 hex digits).

Leaf Registration Methods (inherited usage)
--------------------------------------------

These C++ methods are called from the subclass ``build`` override via ``self.ptr``:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``add_leaf(key: bytes, leaf: bytes)``
     - Map a property name *key* to a ROOT branch suffix *leaf*.  Call
       ``apply_type_prefix()`` after all leaves are registered to prepend
       the ``Type`` prefix.
   * - ``apply_type_prefix()``
     - Prepend ``Type`` to every registered leaf name.  After this call a
       leaf registered as ``b"_pt"`` becomes ``b"top_pt"`` when
       ``Type == "top"``.

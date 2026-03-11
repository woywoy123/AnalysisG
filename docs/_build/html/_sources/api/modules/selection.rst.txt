Selection Template
==================

``selection_template`` is the abstract base class for all user-defined event
selections.  It provides the framework integration (``clone``, ``build``,
``CompileEvent``, ``merge``) and a rich library of typed helper templates for
summing, vectorising, upcasting/downcasting, and uniquifying particle collections
without manual memory management.

Class: ``selection_template``
------------------------------

**Header:** ``<templates/selection_template.h>``

**Inheritance:** ``tools``

Properties
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 20 62

   * - Property
     - Type
     - Description
   * - ``name``
     - ``std::string``
     - Selection class name.  Settable/Gettable.
   * - ``hash``
     - ``std::string``
     - 18-character event hash.  Settable/Gettable.
   * - ``tree``
     - ``std::string``
     - ROOT tree name.  Read-only.
   * - ``weight``
     - ``double``
     - Event weight.  Settable/Gettable.
   * - ``index``
     - ``long``
     - Event sequential index.  Settable.

Public Fields
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Field
     - Type
     - Description
   * - ``passed_weights``
     - ``std::map<std::string, std::map<std::string, float>>``
     - Maps selection-name → {weight-name → value} for passed events.
   * - ``matched_meta``
     - ``std::map<std::string, meta_t>``
     - Maps file hash → metadata struct for this selection.
   * - ``meta_data``
     - ``meta*``
     - Pointer to the dataset metadata object.
   * - ``filename``
     - ``std::string``
     - Source ROOT file path.
   * - ``threadIdx``
     - ``int``
     - Worker-thread index (set by the framework).  Default ``-1``.
   * - ``data``
     - ``event_t``
     - Raw event-level data struct.

Virtual Methods (Override in Subclass)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Signature
     - Description
   * - ``virtual selection_template* clone()``
     - Returns a heap-allocated copy of the selection.
   * - ``virtual bool selection(event_template* ev)``
     - **Primary override.** Return ``true`` to accept the event.
   * - ``virtual bool strategy(event_template* ev)``
     - Secondary filter called before ``selection``.
   * - ``virtual void merge(selection_template* sel)``
     - Merges results from a parallel worker instance *sel* into ``this``.
   * - ``virtual void bulk_write(const long* idx, std::string* hx)``
     - Serialises the selection result for event at index *idx* / hash *hx*.
   * - ``virtual void write(std::vector<particle_template*>* particles, std::string _name, particle_enum attrs)``
     - Writes the kinematic attributes *attrs* for *particles* to the output
       ROOT tree under branch *name*.

Scalar and Particle Write Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Signature
     - Description
   * - ``template<g> void write(g* var, std::string name)``
     - Writes a pointer-to-scalar *var* to the output tree under branch *name*.
   * - ``template<g> void write(g var, std::string name)``
     - Writes a scalar *var* by value to the output tree under branch *name*.
   * - ``template<g> void write(std::vector<g*>* particles, std::string name, particle_enum attrs)``
     - Typed wrapper: downcasts *particles* to ``particle_template*`` and calls
       the virtual ``write``.
   * - ``void switch_board(particle_enum attrs, particle_template* ptr, ...)``
     - Dispatches kinematic attribute extraction to the appropriate output
       vector (``std::vector<double>``, ``std::vector<int>``, …).

Particle Utility Templates
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Signature
     - Description
   * - ``template<g,k> void sum(std::vector<g*>* ch, k** out)``
     - Sums the 4-momenta of *ch* into a new synthesised particle ``*out``.
       Deduplicates by hash.  The result is owned by the selection (freed at
       the end of the event via the internal ``garbage`` map).
   * - ``template<g> g* sum(std::map<std::string,g*>* ch)``
     - Convenience wrapper: vectorises the map and calls the vector overload.
   * - ``template<g> float sum(std::vector<g*>* ch)``
     - Returns the invariant mass (in GeV) of the combined particle.
   * - ``template<g> void safe_delete(std::vector<g*>* particles)``
     - Deletes particles that are **not** marked (``_is_marked == false``),
       i.e. particles that are not synthesised sums.
   * - ``template<g> std::vector<g*> vectorize(std::map<std::string,g*>* in)``
     - Converts a hash→particle map to a flat vector.
   * - ``template<g> std::vector<g*> make_unique(std::vector<g*>* inpt)``
     - Returns a copy of *inpt* with duplicate hashes removed (last occurrence kept).
   * - ``template<g> void downcast(std::vector<g*>* inpt, std::vector<particle_template*>* out)``
     - Appends all elements of *inpt* cast to ``particle_template*`` into *out*.
   * - ``template<o,g> void upcast(std::map<std::string,o*>* inpt, std::vector<g*>* out)``
     - Appends all map values cast to *g* \* into *out*.
   * - ``template<o,g> void upcast(std::vector<o*>* inpt, std::vector<g*>* out)``
     - Appends all vector elements cast to *g* \* into *out*.
   * - ``template<g> void get_leptonics(std::map<std::string,g*> inpt, std::vector<particle_template*>* out)``
     - Filters *inpt* to particles with ``is_lep == true`` or ``is_nu == true``
       and appends them (as ``particle_template*``) to *out*.
   * - ``template<g,j> bool contains(std::vector<g*>* inpt, j* pcheck)``
     - Returns ``true`` if any element of *inpt* has the same hash as *pcheck*.

Framework Methods
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Signature
     - Description
   * - ``bool CompileEvent()``
     - Framework-internal: calls ``strategy`` then ``selection``; manages event
       bookkeeping and bulk-write preparation.
   * - ``selection_template* build(event_template* ev)``
     - Framework-internal: links the selection to the event and calls
       ``CompileEvent``.
   * - ``bool operator == (selection_template& p)``
     - Hash-equality comparison.
   * - ``std::vector<std::map<std::string, float>> reverse_hash(std::vector<std::string>* hashes)``
     - Looks up stored event weights by hash; returns a vector of weight maps.

.. _selection_template_h:

selection_template.h
====================

**@brief** Defines the selection_template class, a base class for event selection strategies.

This file contains the declaration of the selection_template class, which serves as an
abstract base class for implementing specific event selection algorithms in particle physics analysis.
It provides common functionalities like handling event data, managing particle properties,
writing output, and merging results. It acts as a blueprint for concrete selection implementations,
ensuring a consistent interface and providing reusable tools and data structures.

.. cpp:class:: selection_template : public tools

    Abstract base class for defining event selection algorithms and analysis strategies.

    This class provides the fundamental structure and common utilities for event selection
    in a particle physics analysis framework. It inherits from the ``tools`` class, gaining
    access to various helper functions (like hashing).

    Key responsibilities include:
     - Defining a standard interface for selection (``selection``) and subsequent analysis (``strategy``).
     - Managing event-level information (name, hash, weight, index) through ``cproperty`` members.
     - Providing mechanisms for cloning (``clone``) and merging (``merge``) selection instances, crucial for parallel processing.
     - Offering methods for writing particle data (``write``) and other variables to an output format (e.g., ROOT TTree).
     - Handling particle management, including summation (``sum``) and safe deletion (``safe_delete``).
     - Facilitating type casting between base and derived particle types (``downcast``, ``upcast``).
     - Managing metadata and tracking processed events (``passed_weights``, ``matched_meta``).

    Derived classes are expected to override the pure virtual or virtual functions (``selection``,
    ``strategy``, ``clone``, ``merge``, ``bulk_write``) to implement specific analysis logic.

    .. cpp:function:: selection_template()

        Default constructor for selection_template.
        Initializes the ``cproperty`` members (``name``, ``hash``, ``tree``, ``weight``, ``index``)
        by associating them with the current object instance and setting their respective
        getter and setter functions. It also assigns a default name ("selection-template")
        to the ``name`` property.

    .. cpp:function:: virtual ~selection_template()

        Virtual destructor for selection_template.
        Ensures proper cleanup when derived class objects are deleted through a base class pointer.
        Specifically, it iterates through the ``garbage`` map, which stores dynamically allocated
        particles created during operations like ``sum``. It deletes particles that are marked
        (``_is_marked`` is true), preventing memory leaks. Finally, it clears the ``garbage`` map.

    **Properties**

    .. cpp:member:: cproperty<std::string, selection_template> name

        Property representing the descriptive name of the selection strategy.
        This ``cproperty`` allows controlled access (get/set) to the underlying ``data.name`` string.
        It's used to identify the specific analysis or selection being performed.

        :see: :cpp:func:`set_name`, :cpp:func:`get_name`

    .. cpp:function:: static void set_name(std::string* name, selection_template* ev)

        Static setter function callback for the 'name' property.
        Assigns the value pointed to by ``name`` to the ``data.name`` member of the ``selection_template`` instance ``ev``.

        :param name: Pointer to the std::string containing the new name.
        :param ev: Pointer to the selection_template instance whose name is being set.

    .. cpp:function:: static void get_name(std::string* name, selection_template* ev)

        Static getter function callback for the 'name' property.
        Copies the ``data.name`` member from the ``selection_template`` instance ``ev`` into the string pointed to by ``name``.

        :param name: Pointer to the std::string where the retrieved name will be stored.
        :param ev: Pointer to the selection_template instance whose name is being retrieved.

    .. cpp:member:: cproperty<std::string, selection_template> hash

        Property representing a unique hash identifier for the processed event.
        This ``cproperty`` manages access to the ``data.hash`` string. The hash is typically
        generated based on the input file path and the event's index within that file,
        providing a unique way to reference a specific event across different processing stages.

        :see: :cpp:func:`set_hash`, :cpp:func:`get_hash`

    .. cpp:function:: static void set_hash(std::string* path, selection_template* ev)

        Static setter function callback for the 'hash' property.
        If the hash for the event ``ev`` is not already set (``ev->data.hash`` is empty),
        it generates a unique hash. The hash is created by concatenating the string pointed
        to by ``path`` (typically the input filename) and the event index (``ev->index``),
        then applying the ``tools::hash`` function. The input ``path`` string is cleared after use.
        If the hash is already set, this function does nothing.

        :param path: Pointer to the std::string containing the base path (e.g., filename) for hash generation. This string is cleared by the function.
        :param ev: Pointer to the selection_template instance whose hash is being set.

    .. cpp:function:: static void get_hash(std::string* val, selection_template* ev)

        Static getter function callback for the 'hash' property.
        Copies the ``data.hash`` member from the ``selection_template`` instance ``ev`` into the string pointed to by ``val``.

        :param val: Pointer to the std::string where the retrieved hash will be stored.
        :param ev: Pointer to the selection_template instance whose hash is being retrieved.

    .. cpp:member:: cproperty<std::string, selection_template> tree

        Property representing the name of the output tree (e.g., ROOT TTree) associated with this selection.
        This ``cproperty`` provides read-only access (getter only) to the ``data.tree`` string.
        It specifies the destination within the output file where results from this selection strategy should be stored.

        :see: :cpp:func:`get_tree`

    .. cpp:function:: static void get_tree(std::string* name, selection_template* ev)

        Static getter function callback for the 'tree' property.
        Copies the ``data.tree`` member from the ``selection_template`` instance ``ev`` into the string pointed to by ``name``.

        :param name: Pointer to the std::string where the retrieved tree name will be stored.
        :param ev: Pointer to the selection_template instance whose tree name is being retrieved.

    .. cpp:member:: cproperty<double, selection_template> weight

        Property representing the weight associated with the event.
        This ``cproperty`` allows controlled access (get/set) to the ``data.weight`` double.
        Event weights are crucial in physics analysis for accounting for factors like cross-sections,
        efficiencies, and luminosity.

        :see: :cpp:func:`set_weight`, :cpp:func:`get_weight`

    .. cpp:function:: static void set_weight(double* inpt, selection_template* ev)

        Static setter function callback for the 'weight' property.
        Assigns the value pointed to by ``inpt`` to the ``data.weight`` member of the ``selection_template`` instance ``ev``.

        :param inpt: Pointer to the double containing the new event weight.
        :param ev: Pointer to the selection_template instance whose weight is being set.

    .. cpp:function:: static void get_weight(double* inpt, selection_template* ev)

        Static getter function callback for the 'weight' property.
        Copies the ``data.weight`` member from the ``selection_template`` instance ``ev`` into the double pointed to by ``inpt``.

        :param inpt: Pointer to the double where the retrieved weight will be stored.
        :param ev: Pointer to the selection_template instance whose weight is being retrieved.

    .. cpp:member:: cproperty<long, selection_template> index

        Property representing the index (entry number) of the event within its original input file.
        This ``cproperty`` allows controlled access (setter only, getter implicitly via ``data.index``) to the ``data.index`` long.
        The index, combined with the filename, helps uniquely identify an event.

        :see: :cpp:func:`set_index`

    .. cpp:function:: static void set_index(long* inpt, selection_template* ev)

        Static setter function callback for the 'index' property.
        Assigns the value pointed to by ``inpt`` to the ``data.index`` member of the ``selection_template`` instance ``ev``.

        :param inpt: Pointer to the long containing the new event index.
        :param ev: Pointer to the selection_template instance whose index is being set.

    **Core Virtual Methods (to be overridden by derived classes)**

    .. cpp:function:: virtual selection_template* clone()

        Virtual function to create a deep copy (clone) of the current selection_template object.
        This is essential for parallel processing, where each thread or process needs its own
        independent instance of the selection logic. Derived classes MUST override this method
        to return a new instance of their specific type, ensuring correct polymorphism.
        The base implementation returns a new ``selection_template`` instance, which is usually
        insufficient for derived classes.

        :return: A pointer to a newly allocated selection_template object that is a copy of the current one. The caller is responsible for managing the memory of the returned object.

    .. cpp:function:: virtual bool selection(event_template* ev)

        Virtual function defining the primary event selection logic.
        Derived classes MUST implement this function to apply their specific selection criteria
        to the given event ``ev``. This function typically checks particle properties, event kinematics,
        topology, etc., to decide if the event is of interest.
        The base implementation simply returns true.

        :param ev: Pointer to the :cpp:class:`event_template` object representing the current event to be evaluated.
        :return: ``true`` if the event passes the selection criteria, ``false`` otherwise.

    .. cpp:function:: virtual bool strategy(event_template* ev)

        Virtual function defining the analysis strategy or further processing steps after an event passes selection.
        Derived classes MUST implement this function to perform tasks on events that have already
        passed the ``selection`` stage. This might include calculating derived quantities, filling histograms,
        reconstructing particles, or preparing data for output.
        The base implementation simply returns true.

        :param ev: Pointer to the :cpp:class:`event_template` object that has passed the ``selection`` criteria.
        :return: ``true`` if the strategy execution is successful, ``false`` otherwise (e.g., if an error occurs during processing).

    .. cpp:function:: virtual void merge(selection_template* sel)

        Virtual function to merge data from another selection_template instance into this one.
        Derived classes MUST implement this function to define how results (e.g., histograms, counters,
        accumulated statistics) from different instances are combined. This is crucial for aggregating
        results from parallel processing or sequential file processing. The base implementation does nothing.

        :param sel: Pointer to the selection_template object whose data should be merged into the current object. The state of ``sel`` might be modified or considered consumed after merging, depending on the implementation.

    .. cpp:function:: virtual void bulk_write(const long* idx, std::string* hx)

        Virtual function for potentially optimized bulk writing of minimal event information (index and hash).
        This function is intended for scenarios where only the index and hash of passed events need to be
        recorded efficiently, potentially bypassing more complex output structures. Derived classes can
        override this to implement specific bulk writing logic. Calling the base implementation directly
        disables further bulk writing by setting ``p_bulk_write`` to false.

        :param idx: Pointer to the event index (``long``) to be written.
        :param hx: Pointer to the event hash (``std::string``) to be written.

    **Output Methods**

    .. cpp:function:: virtual void write(std::vector<particle_template*>* particles, std::string name, particle_enum attrs)

        Writes a specific attribute of a collection of particles to the output.
        This function iterates through the provided vector of ``particles``, extracts the specified ``attrs``
        attribute from each particle using the ``switch_board`` helper functions, collects the values
        into a temporary vector (e.g., ``std::vector<double>`` for pt), and then calls the appropriate
        templated ``write`` function to persist this vector as a branch in the output file (e.g., TTree).
        The branch name is constructed by appending a suffix (e.g., "_pt") to the provided ``name``.

        :param particles: Pointer to a vector of :cpp:class:`particle_template` pointers whose attribute should be written.
        :param name: The base name for the output branch. The attribute suffix will be appended automatically.
        :param attrs: The ``particle_enum`` value specifying which particle attribute to extract and write (e.g., ``particle_enum::pt``, ``particle_enum::pdgid``).

    .. cpp:function:: template <typename g> void write(g* var, std::string name)

        Template function to write the value of a variable (passed by pointer) to the output tree.
        This function assumes an output mechanism (like a ROOT TTree, represented by the ``handle``) is available.
        It creates a branch named ``name`` in the output and writes the data pointed to by ``var``.
        The type ``g`` determines the data type of the branch. Requires ``handle`` to be initialized.

        :tparam g: The data type of the variable to write.
        :param var: Pointer to the variable whose content needs to be written.
        :param name: The desired name for the branch in the output structure.

        .. note:: The ``handle`` member must be pointing to a valid output object (e.g., ``write_t*``).

    .. cpp:function:: template <typename g> void write(g var, std::string name)

        Template function to write the value of a variable (passed by value) to the output tree.
        Similar to the pointer version, this function writes the value of ``var`` to a branch named ``name``.
        It's suitable for writing single values per event (like event weight, number of jets, etc.).
        Requires ``handle`` to be initialized.

        :tparam g: The data type of the variable to write.
        :param var: The variable value to be written.
        :param name: The desired name for the branch in the output structure.

        .. note:: The ``handle`` member must be pointing to a valid output object (e.g., ``write_t*``).

    **Helper and Utility Methods**

    .. cpp:function:: void switch_board(particle_enum attrs, particle_template* ptr, std::vector<int>* data)

        Helper function (overload) to populate integer data vectors based on particle attributes.
        This function is called internally by the ``write`` method. Based on the ``attrs`` enum, it extracts
        the corresponding integer attribute (e.g., PDG ID, index) from the :cpp:class:`particle_template` pointed to by ``ptr``
        and appends it to the ``data`` vector.

        :param attrs: The particle attribute (``particle_enum``) to extract (must correspond to an integer type like ``pdgid`` or ``index``).
        :param ptr: Pointer to the :cpp:class:`particle_template` object from which to extract the data.
        :param data: Pointer to the ``std::vector<int>`` where the extracted integer value will be added.

    .. cpp:function:: void switch_board(particle_enum attrs, particle_template* ptr, std::vector<double>* data)

        Helper function (overload) to populate double data vectors based on particle attributes.
        This function is called internally by the ``write`` method. Based on the ``attrs`` enum, it extracts
        the corresponding double attribute (e.g., pt, eta, phi, energy, mass, charge, px, py, pz) from the
        :cpp:class:`particle_template` pointed to by ``ptr`` and appends it to the ``data`` vector.

        :param attrs: The particle attribute (``particle_enum``) to extract (must correspond to a double type like ``pt``, ``eta``, ``energy``, etc.).
        :param ptr: Pointer to the :cpp:class:`particle_template` object from which to extract the data.
        :param data: Pointer to the ``std::vector<double>`` where the extracted double value will be added.

    .. cpp:function:: void switch_board(particle_enum attrs, particle_template* ptr, std::vector<bool>* data)

        Helper function (overload) to populate boolean data vectors based on particle attributes.
        This function is called internally by the ``write`` method. Based on the ``attrs`` enum, it extracts
        the corresponding boolean attribute (e.g., is_b, is_lep, is_nu, is_add) from the :cpp:class:`particle_template`
        pointed to by ``ptr`` and appends it to the ``data`` vector.

        :param attrs: The particle attribute (``particle_enum``) to extract (must correspond to a boolean type like ``is_b``, ``is_lep``, etc.).
        :param ptr: Pointer to the :cpp:class:`particle_template` object from which to extract the data.
        :param data: Pointer to the ``std::vector<bool>`` where the extracted boolean value will be added.

    .. cpp:function:: void switch_board(particle_enum attrs, particle_template* ptr, std::vector<std::vector<double>>* data)

        Helper function (overload) to populate vector-of-vector-of-double data based on particle attributes.
        This function is called internally by the ``write`` method for composite attributes like four-vectors.
        Based on the ``attrs`` enum (e.g., ``pmc`` for (px, py, pz, E), ``pmu`` for (pt, eta, phi, E)), it constructs
        a ``std::vector<double>`` containing the components from the :cpp:class:`particle_template` pointed to by ``ptr``
        (using calls to the double ``switch_board``) and appends this vector to the ``data`` vector.

        :param attrs: The particle attribute (``particle_enum``) to extract (must correspond to a composite type like ``pmc`` or ``pmu``).
        :param ptr: Pointer to the :cpp:class:`particle_template` object from which to extract the data.
        :param data: Pointer to the ``std::vector<std::vector<double>>`` where the extracted vector<double> will be added.

    .. cpp:function:: std::vector<std::map<std::string, float>> reverse_hash(std::vector<std::string>* hashes)

        Reverses event hashes to find their corresponding original file names and weights.
        This function takes a list of event hashes (``hashes``) and searches through the ``passed_weights``
        map (which stores weights of selected events, indexed by filename and then hash). For each input hash,
        it finds the corresponding entry in ``passed_weights`` and returns the filename and weight.

        :param hashes: Pointer to a ``std::vector<std::string>`` containing the event hashes to look up.
        :return: A ``std::vector`` of ``std::map<std::string, float>``. Each map in the vector corresponds to an input hash in the same order. If a hash is found, the map contains ``{filename, weight}``. If a hash is not found, the map contains ``{"None", 0}``.

    .. cpp:function:: bool CompileEvent()

        Executes the core event processing logic: selection and strategy.
        This method first calls the virtual ``selection`` function with the currently associated
        event (``m_event``). If ``selection`` returns ``true``, it then calls the virtual ``strategy``
        function, also with ``m_event``.

        :return: ``true`` if *both* ``selection(m_event)`` and ``strategy(m_event)`` return ``true``. Returns ``false`` if ``selection`` returns ``false``, or if ``selection`` returns ``true`` but ``strategy`` returns ``false``.
        :note: This method requires ``m_event`` to be non-null, meaning it should be called on an instance configured for a specific event (typically created via ``build``).

    .. cpp:function:: selection_template* build(event_template* ev)

        Creates and configures a new selection instance tailored for a specific event.
        This method first calls ``clone()`` to create a new instance of the (derived) selection class.
        It then associates this new instance with the provided :cpp:class:`event_template` (``ev``) by setting ``m_event``.
        It copies relevant event data (``ev->data``) and properties (``ev->weight``, ``ev->filename``)
        into the new selection instance. The original ``name`` of the selection strategy is preserved.

        :param ev: Pointer to the :cpp:class:`event_template` object representing the event to be processed by the new instance.
        :return: A pointer to the newly created and configured ``selection_template`` instance. The caller might be responsible for managing the memory of this object, depending on the framework's design.

    .. cpp:function:: bool operator==(selection_template& p)

        Equality comparison operator based on event hash.
        Compares the ``hash`` property of the current ``selection_template`` object with the ``hash``
        property of the ``selection_template`` object ``p``.

        :param p: The ``selection_template`` object to compare against.
        :return: ``true`` if the ``hash`` values of both objects are identical, ``false`` otherwise.

    **Particle Manipulation Templates**

    .. cpp:function:: template <typename g, typename k> void sum(std::vector<g*>* ch, k** out)

        Template function to calculate the vector sum of the four-momenta of a list of particles.
        Creates a new particle of type ``k`` whose four-momentum is the sum of the four-momenta
        of the unique particles in the input vector ``ch``. Uniqueness is determined by particle hash
        to avoid double-counting. The newly created summed particle (``*out``) is registered in the
        ``garbage`` collection map for automatic memory management.

        :tparam g: The type of the input particles in the vector (must inherit from :cpp:class:`particle_template`).
        :tparam k: The type of the output summed particle (must inherit from :cpp:class:`particle_template`).
        :param ch: Pointer to a ``std::vector`` of pointers to the input particles (``g*``).
        :param out: Pointer to a pointer (``k**``) where the address of the newly created summed particle (``k*``) will be stored. The particle pointed to by ``*out`` after the call is managed by the ``garbage`` collector.

    .. cpp:function:: template <typename g> void safe_delete(std::vector<g*>* particles)

        Template function to safely delete particles within a vector, respecting ownership flags.
        Iterates through the ``particles`` vector. For each particle pointer, it checks if the particle
        is *not* marked (``_is_marked`` is false). If it's not marked, the particle object is deleted,
        and the corresponding pointer in the vector is set to ``nullptr``. Marked particles are assumed
        to be managed elsewhere (e.g., by the ``garbage`` collector or the original event data) and are not deleted.

        :tparam g: The type of particles in the vector (must inherit from :cpp:class:`particle_template`).
        :param particles: Pointer to the ``std::vector`` of particle pointers (``g*``) to clean up. Pointers to deleted particles within this vector will be set to ``nullptr``.

    .. cpp:function:: template <typename g> g* sum(std::map<std::string, g*>* ch)

        Template function to calculate the vector sum of the four-momenta of particles stored in a map.
        This function first converts the input map ``ch`` (mapping hash strings to particle pointers)
        into a ``std::vector`` of particle pointers. It then calls the vector-based ``sum`` function
        (the overload calculating invariant mass) on this temporary vector.

        :tparam g: The type of particles stored as values in the map (must inherit from :cpp:class:`particle_template`).
        :param ch: Pointer to a ``std::map`` where keys are ``std::string`` (hashes) and values are particle pointers (``g*``).
        :return: A pointer to the newly created summed particle (``g*``). The particle is managed by the ``garbage`` collector. Returns ``nullptr`` if the input map ``ch`` is empty.
        :note: This overload seems to return a ``g*`` representing the summed particle, not its mass, contrasting with the float-returning ``sum`` overload.

    .. cpp:function:: template <typename g> float sum(std::vector<g*>* ch)

        Template function to calculate the invariant mass of the sum of particles in a vector.
        This function internally calls the ``sum<g, particle_template>`` template function to compute the
        four-vector sum of the particles in ``ch``, creating a temporary :cpp:class:`particle_template` object
        (managed by the ``garbage`` collector). It then returns the invariant mass (``mass``) of this summed particle,
        typically in GeV.

        :tparam g: The type of particles in the input vector (must inherit from :cpp:class:`particle_template`).
        :param ch: Pointer to a ``std::vector`` of pointers to the input particles (``g*``).
        :return: The invariant mass (float) of the combined system represented by the particles in ``ch``, in units of GeV.

    **Particle Container Manipulation Templates**

    .. cpp:function:: template <typename g> std::vector<g*> vectorize(std::map<std::string, g*>* in)

        Template function to convert a map of particles (hash -> pointer) into a vector of particle pointers.
        Iterates through the key-value pairs in the input map ``in`` and creates a ``std::vector`` containing
        only the particle pointers (the map's values). The order in the resulting vector depends on the map's iteration order.

        :tparam g: The type of particles stored as values in the map.
        :param in: Pointer to the input ``std::map<std::string, g*>`` to convert.
        :return: A ``std::vector<g*>`` containing all the particle pointers from the input map's values.

    .. cpp:function:: template <typename g> std::vector<g*> make_unique(std::vector<g*>* inpt)

        Template function to create a vector containing only unique particles from an input vector.
        Iterates through the input vector ``inpt``. For each particle, it checks if a particle with the
        same hash already exists in the output vector being built. If not, the particle pointer is added
        to the output vector. This effectively removes duplicate particles based on their hash identifier.

        :tparam g: The type of particles in the vectors.
        :param inpt: Pointer to the input ``std::vector<g*>`` which may contain duplicate particles (by hash).
        :return: A ``std::vector<g*>`` containing only the unique particle pointers from the input vector. The relative order of the first occurrences of unique particles might be preserved.

    .. cpp:function:: template <typename g> void downcast(std::vector<g*>* inpt, std::vector<particle_template*>* out)

        Template function to perform a static downcast on a vector of derived particle pointers to a vector of base particle pointers.
        Iterates through the input vector ``inpt`` containing pointers of derived type ``g``. For each pointer,
        it performs a ``static_cast`` to the base type ``particle_template*`` and adds the resulting pointer
        to the output vector ``out``. This is safe assuming ``g`` publicly inherits from :cpp:class:`particle_template`.

        :tparam g: The derived particle type (must inherit from :cpp:class:`particle_template`).
        :param inpt: Pointer to the input ``std::vector<g*>`` containing pointers to derived objects.
        :param out: Pointer to the output ``std::vector<particle_template*>`` where the base class pointers will be added.

    .. cpp:function:: template <typename o, typename g> void upcast(std::map<std::string, o*>* inpt, std::vector<g*>* out)

        Template function to perform a dynamic upcast on particles from a map of base types to a vector of derived types.
        Iterates through the input map ``inpt`` containing pointers of base type ``o`` (e.g., :cpp:class:`particle_template`).
        For each pointer, it attempts a ``dynamic_cast`` to the derived type ``g*``. If the cast is successful
        (i.e., the object is actually of type ``g`` or a further derived type), the resulting derived pointer
        is added to the output vector ``out``. If the cast fails, nothing is added for that particle.

        :tparam o: The base particle type (e.g., :cpp:class:`particle_template`).
        :tparam g: The desired derived particle type.
        :param inpt: Pointer to the input ``std::map<std::string, o*>`` containing pointers to base objects.
        :param out: Pointer to the output ``std::vector<g*>`` where successfully cast derived pointers will be added.

    .. cpp:function:: template <typename o, typename g> void upcast(std::vector<o*>* inpt, std::vector<g*>* out)

        Template function to perform a dynamic upcast on particles from a vector of base types to a vector of derived types.
        Iterates through the input vector ``inpt`` containing pointers of base type ``o`` (e.g., :cpp:class:`particle_template`).
        For each pointer, it attempts a ``dynamic_cast`` to the derived type ``g*``. If the cast is successful
        (i.e., the object is actually of type ``g`` or a further derived type), the resulting derived pointer
        is added to the output vector ``out``. If the cast fails, nothing is added for that particle.

        :tparam o: The base particle type (e.g., :cpp:class:`particle_template`).
        :tparam g: The desired derived particle type.
        :param inpt: Pointer to the input ``std::vector<o*>`` containing pointers to base objects.
        :param out: Pointer to the output ``std::vector<g*>`` where successfully cast derived pointers will be added.

    .. cpp:function:: template <typename g> void get_leptonics(std::map<std::string, g*> inpt, std::vector<particle_template*>* out)

        Template function to extract particles identified as leptons or neutrinos from a map.
        Iterates through the input map ``inpt``. For each particle pointer (``g*``), it checks if the particle's
        ``is_lep`` or ``is_nu`` flag is true. If either flag is true, the particle pointer (cast to ``particle_template*``)
        is added to the output vector ``out``.

        :tparam g: The particle type stored as values in the map (must inherit from :cpp:class:`particle_template`).
        :param inpt: The input ``std::map<std::string, g*>`` containing particles to check.
        :param out: Pointer to the output ``std::vector<particle_template*>`` where pointers to identified leptonic particles will be added.

    .. cpp:function:: template <typename g, typename j> bool contains(std::vector<g*>* inpt, j* pcheck)

        Template function to check if a specific particle (by hash) exists within a vector of particles.
        Iterates through the input vector ``inpt``. For each particle ``g*`` in the vector, it compares its
        hash (``->hash``) with the hash of the particle ``pcheck`` (``->hash``).

        :tparam g: The type of particles in the vector ``inpt``.
        :tparam j: The type of the particle ``pcheck`` to search for. Both ``g`` and ``j`` must have a ``hash`` member or method accessible.
        :param inpt: Pointer to the ``std::vector<g*>`` to search within.
        :param pcheck: Pointer to the particle (``j*``) whose presence (based on hash) is being checked.
        :return: ``true`` if a particle ``p`` in ``inpt`` exists such that ``p->hash == pcheck->hash``, ``false`` otherwise.

    **Public Member Variables**

    .. cpp:member:: meta* meta_data = nullptr

        Pointer to the metadata object associated with the current analysis run or dataset.
        Provides access to global information like luminosity, cross-sections, sample type, etc.
        Initialized externally, typically by the framework managing the selection instance. Defaults to ``nullptr``.

    .. cpp:member:: std::string filename = ""

        Stores the filename (including path) of the input file currently being processed by this instance.
        Used for tracking event origins, generating hashes, and organizing results (e.g., in ``passed_weights``).

    .. cpp:member:: event_t data

        Structure holding the core event identification data (name, hash, tree, weight, index).
        The ``cproperty`` members provide controlled access to the fields within this structure.

    .. cpp:member:: int threadIdx = -1

        Index of the processing thread assigned to this selection instance.
        Used in multi-threaded environments to potentially manage thread-specific resources or logging.
        Defaults to -1, indicating single-threaded operation or unassigned thread.

    .. cpp:member:: std::map<std::string, std::map<std::string, float>> passed_weights = {}

        Stores the weights of events that have passed the selection criteria.
        The outer map is keyed by the input filename (``std::string``).
        The inner map is keyed by the event hash (``std::string``) and stores the event weight (``float``).
        This structure is populated during the merging process (``merger``) when combining results
        from individual event processing instances.

    .. cpp:member:: std::map<std::string, meta_t> matched_meta = {}

        Map storing metadata associated with processed input files.
        Keyed by the input filename (``std::string``), the value is a ``meta_t`` structure
        (presumably containing metadata extracted from or relevant to that file).
        Populated during the merging process (``merger``) from the ``meta_data`` of individual event instances.

    **Private Members**
    *(Note: Private members are typically not included in public API documentation, but listed here for completeness based on the input)*

    .. cpp:function:: void bulk_write_out()

        Performs the actual bulk writing operation if enabled and configured.
        *(Private member)*

    .. cpp:function:: void merger(selection_template* sl2)

        Internal merging logic, typically called by the managing ``container``.
        *(Private member)*

    .. cpp:member:: std::unordered_map<long, std::string> sequence

        Map storing the sequence of event indices and hashes for bulk writing.
        *(Private member)*

    .. cpp:member:: bool p_bulk_write = true

        Flag indicating whether bulk writing mode is enabled.
        *(Private member)*

    .. cpp:member:: write_t* handle = nullptr

        Pointer to the underlying output writing mechanism.
        *(Private member)*

    .. cpp:member:: event_template* m_event = nullptr

        Pointer to the specific event being processed by this instance. Null if accumulator.
        *(Private member)*

    .. cpp:member:: std::map<std::string, std::vector<particle_template*>> garbage = {}

        Garbage collection map for dynamically created particles.
        *(Private member)*


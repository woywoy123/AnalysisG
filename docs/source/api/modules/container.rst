Container Module
================

The ``container`` class manages per-file typed data containers used by
``sampletracer`` to store events, graphs, selections, and metadata for each
ROOT/HDF5 file.  One ``container`` instance is created per (file, label) pair.

Struct: ``entry_t``
--------------------

**Header:** ``<container/container.h>``

``entry_t`` is the fundamental storage record; each unique event hash maps to
exactly one ``entry_t``.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Field
     - Type
     - Description
   * - ``hash``
     - ``std::string``
     - 18-character event hash (key in the ``container::random_access`` map).
   * - ``m_data``
     - ``std::vector<graph_t*>``
     - Compiled ``graph_t`` objects ready for the dataloader.
   * - ``m_graph``
     - ``std::vector<graph_template*>``
     - Compiled ``graph_template`` objects.
   * - ``m_event``
     - ``std::vector<event_template*>``
     - Compiled ``event_template`` objects.
   * - ``m_selection``
     - ``std::vector<selection_template*>``
     - Compiled ``selection_template`` results.

``entry_t`` Methods:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Signature
     - Description
   * - ``void init()``
     - Initialises the slot (called when the slot is first created).
   * - ``void destroy()``
     - Frees all stored object pointers.
   * - ``bool has_event(event_template* ev)``
     - Returns ``true`` if *ev* (by hash) is already in ``m_event``.
   * - ``bool has_graph(graph_template* gr)``
     - Returns ``true`` if *gr* is already in ``m_graph``.
   * - ``bool has_selection(selection_template* sel)``
     - Returns ``true`` if *sel* is already in ``m_selection``.
   * - ``template<g> void destroy(std::vector<g*>* c)``
     - Deletes and clears each element in *c*.

Class: ``container``
---------------------

**Header:** ``<container/container.h>``

**Inheritance:** ``tools``

Public Fields
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 28 44

   * - Field
     - Type
     - Description
   * - ``meta_data``
     - ``meta*``
     - Pointer to the dataset metadata object (one per file, shared).
   * - ``filename``
     - ``std::string*``
     - Pointer to the source ROOT/HDF5 file path.
   * - ``output_path``
     - ``std::string*``
     - Pointer to the output directory path.
   * - ``label``
     - ``std::string``
     - Dataset label (as registered with ``analysis::add_samples``).
   * - ``random_access``
     - ``std::map<std::string, entry_t>``
     - Maps event hash → ``entry_t`` slot.
   * - ``merged``
     - ``std::map<std::string, selection_template*>*``
     - Pointer to the merged selection map (set by ``sampletracer``).

Public Methods
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Signature
     - Description
   * - ``void add_meta_data(meta*, std::string)``
     - Sets ``meta_data`` and associates it with the given filename.
   * - ``meta* get_meta_data()``
     - Returns ``meta_data``.
   * - ``bool add_event_template(event_template*, std::string label)``
     - Adds a compiled event to the slot for its hash.  Returns ``false``
       if already present.
   * - ``bool add_graph_template(graph_template*, std::string label)``
     - Adds a compiled graph.  Returns ``false`` if already present.
   * - ``bool add_selection_template(selection_template*)``
     - Adds a selection result.  Returns ``false`` if already present.
   * - ``void fill_selections(std::map<std::string, selection_template*>* inpt)``
     - Populates *inpt* from all selections stored in ``random_access``.
   * - ``void get_events(std::vector<event_template*>*, std::string label)``
     - Appends events with matching *label* to the output vector.
   * - ``void populate_dataloader(dataloader* dl)``
     - Transfers all ``graph_t*`` objects to the dataloader.
   * - ``void compile(size_t* len, int threadIdx, int thrd)``
     - Worker-thread entry point: compiles all events, graphs, and
       selections assigned to this thread.
   * - ``size_t len()``
     - Returns the total number of entries in ``random_access``.
   * - ``entry_t* add_entry(std::string hash)``
     - Creates (if not already present) and returns the ``entry_t`` for *hash*.

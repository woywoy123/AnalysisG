Sample Tracer Module
====================

The ``sampletracer`` class is the in-memory event store.  It manages metadata,
events, graphs, and selections organised into ``container`` objects (one per
ROOT file/label pair), and provides the bridge from fully compiled event objects
to the ``dataloader``'s graph tensor dataset.

Class: ``sampletracer``
------------------------

**Header:** ``<generators/sampletracer.h>``

**Inheritance:** ``tools``, ``notification``

Public Fields
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Field
     - Type
     - Description
   * - ``output_path``
     - ``std::string*``
     - Pointer to the output directory string (set by ``analysis``).

Public Methods
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Signature
     - Description
   * - ``bool add_meta_data(meta* meta_, std::string filename)``
     - Associates the ``meta`` object *meta_* with *filename*.  Returns
       ``false`` if an entry for *filename* already exists.
   * - ``meta* get_meta_data(std::string filename)``
     - Returns the ``meta*`` associated with *filename*, or ``nullptr``.
   * - ``std::vector<event_template*> get_events(std::string label)``
     - Returns all events stored under dataset label *label*.
   * - ``bool add_event(event_template* ev, std::string label)``
     - Adds a fully compiled event to the container for *label*.
       Returns ``false`` if the event already exists.
   * - ``bool add_graph(graph_template* gr, std::string label)``
     - Adds a compiled graph to the container for *label*.
       Returns ``false`` if it already exists.
   * - ``bool add_selection(selection_template* sel)``
     - Adds a selection result.  Returns ``false`` if it already exists.
   * - ``void fill_selections(std::map<std::string, selection_template*>* inpt)``
     - Populates *inpt* from each ``container``'s stored selections.
   * - ``void populate_dataloader(dataloader* dl)``
     - Transfers all compiled ``graph_t`` objects to the dataloader's
       internal dataset for training.
   * - ``void compile_objects(int threads, int intrath)``
     - Compiles events, graphs, and selections across *threads* parallel
       workers with *intrath* intra-op threads each.

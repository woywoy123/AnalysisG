GraphTemplate (Python)
======================

The ``GraphTemplate`` Cython class wraps the C++ ``graph_template``.
User graph classes must subclass it and override ``CompileEvent`` to call
``define_particle_nodes`` and register features.

Special Methods
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Signature
     - Description
   * - ``__hash__() → int``
     - Integer hash from the first 8 hex digits of ``self.hash``.
   * - ``__eq__(other) → bool``
     - Equality based on C++ ``operator==``.
   * - ``is_self(inpt) → bool``
     - Return ``True`` if ``inpt`` is an instance or subclass of
       ``GraphTemplate``.

Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Property
     - Type
     - Description
   * - ``index``
     - ``int``
     - Event index (read-only, from the underlying ``graph_t``).
   * - ``Tree``
     - ``str``
     - ROOT TTree name (read-only).
   * - ``PreSelection``
     - ``bool``
     - Pre-selection flag (read/write).  Return ``False`` in
       ``CompileEvent`` to discard the graph.

C++ Interface (called from subclass ``CompileEvent``)
------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Method
     - Description
   * - ``define_particle_nodes(ptr)``
     - Set the node particle list from a ``std::vector<ParticleType*>*``.
   * - ``define_topology(fn)``
     - Set the edge connectivity function ``fn(G*, tuple<P*,P*>*)``
       which returns ``True`` when two particles share an edge.
   * - ``get_event[T]() → T*``
     - Return the underlying event cast to type ``T``.
   * - ``add_graph_truth_feature(name, fn)``
     - Register a truth graph-level feature function.
   * - ``add_graph_data_feature(name, fn)``
     - Register a data graph-level feature function.
   * - ``add_node_truth_feature(name, fn)``
     - Register a truth node-level feature function.
   * - ``add_node_data_feature(name, fn)``
     - Register a data node-level feature function.
   * - ``add_edge_truth_feature(name, fn)``
     - Register a truth edge-level feature function.
   * - ``add_edge_data_feature(name, fn)``
     - Register a data edge-level feature function.

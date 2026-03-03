SelectionTemplate (Python)
==========================

The ``SelectionTemplate`` Cython class wraps the C++ ``selection_template``.
User selection classes must subclass it and override ``selection`` (and
optionally ``strategy``, ``Postprocessing``, ``merge``).

Special Methods
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Signature
     - Description
   * - ``__hash__() → int``
     - Hash derived from the event hash string.
   * - ``dump(path, name) / load(path, name)``
     - Pickle-serialise or restore the selection object.
   * - ``HashToWeightFile(hash_) → list[dict[str, float]]``
     - Return the per-variable weight maps for the given event hash.
   * - ``InterpretROOT(path: str, tree: str)``
     - Re-run over a saved selection ROOT output file to populate
       ``root_leaves`` for post-processing.
   * - ``Postprocessing()``
     - Hook called after all events are processed.  Override to produce
       plots or derived quantities.

Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Type
     - Description
   * - ``PassedWeights``
     - ``dict[str, dict[str, float]]``
     - Map of passed event hashes to their per-variable weight maps.
   * - ``GetMetaData``
     - ``dict[str, Meta]``
     - Dictionary of :class:`Meta` objects keyed by dataset label.

C++ Interface
-------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method / field
     - Description
   * - ``selection(ev: EventTemplate) → bool``
     - Override: return ``True`` to accept the event.
   * - ``strategy(ev: EventTemplate)``
     - Override: additional processing for accepted events.
   * - ``merge(other: SelectionTemplate)``
     - Override: merge parallel results into ``self``.
   * - ``write(name, particle_list)``
     - Serialise a list of particle attributes to the output ROOT tree.

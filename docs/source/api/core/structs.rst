Structs (Python)
================

The ``structs`` Cython module exposes the C++ plain-old-data types as
Cython extension types.  They are used internally by the framework and
are not normally instantiated directly by users.

particle_t
----------

Holds per-particle metadata extracted from ROOT:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Type
     - Description
   * - ``index``
     - ``int``
     - Sequential particle index in the event.
   * - ``type``
     - ``bytes``
     - ROOT branch type prefix (e.g. ``b"jet"``).
   * - ``lepdef``
     - ``list[int]``
     - PDG IDs that count as leptons.
   * - ``nudef``
     - ``list[int]``
     - PDG IDs that count as neutrinos.

data_t
------

Holds one ROOT leaf value together with its type tag (``data_enum``).
Used internally by the ``io`` engine to pass typed data to ``build``.

data_enum
---------

Enumerates all supported ROOT leaf data types:

``v_f``, ``v_d``, ``v_i``, ``v_l``, ``v_b``, ``v_c``, ``v_ull``,
``v_ui``, ``vv_f``, ``vv_d``, ``vv_i``, ``vv_l``, ``vv_b``, ``vv_c``,
``vvv_f``, ``vvv_d``, ``vvv_i``, ``vvv_l``, ``vvv_b``.

(prefix ``v`` = vector/list, ``vv`` = nested list, ``vvv`` = doubly nested.)

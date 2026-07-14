Notification (Python)
=====================

The ``Notification`` Cython class is a thin wrapper around the C++
``notification`` class.  It is a base class that provides coloured
console output, verbosity control, and warning/error counting.

The class is not normally instantiated directly.  All framework classes
that inherit from ``notification`` (such as :class:`Analysis`,
:class:`IO`, :class:`ParticleTemplate`, etc.) automatically gain its
logging interface.

C++ ``notification`` Interface
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method / field
     - Description
   * - ``shush`` *(bool)*
     - Set to ``True`` to suppress all console output.
   * - ``info(msg: bytes)``
     - Print an informational message (green prefix).
   * - ``warning(msg: bytes)``
     - Print a warning message (yellow prefix) and increment the warning
       counter.
   * - ``failure(msg: bytes)``
     - Print an error message (red prefix) and increment the failure
       counter.
   * - ``success(msg: bytes)``
     - Print a success message (green tick prefix).
   * - ``progress(done: int, total: int, msg: bytes)``
     - Print an in-place progress bar.

Notification Module
===================

The ``notification`` class provides coloured terminal logging and multi-threaded
progress-bar utilities. Every major framework class (``analysis``, ``io``,
``sampletracer``, ``optimizer``, ``metrics``, ``model_template``, etc.) inherits
from ``notification`` so that consistent console output is available everywhere.

Class: ``notification``
-----------------------

**Header:** ``<notification/notification.h>``

**Inheritance:** (base class — no parents)

Public Member Variables
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Name
     - Type
     - Description
   * - ``prefix``
     - ``std::string``
     - Optional string prepended to every message (e.g. class name).
   * - ``shush``
     - ``bool``
     - When ``true``, suppresses all output.  Default ``false``.
   * - ``bold``
     - ``bool``
     - When ``true``, messages are rendered in bold ANSI text.  Default ``false``.
   * - ``_warning``
     - ``int``
     - ANSI colour code for warnings.  Default ``33`` (yellow).
   * - ``_failure``
     - ``int``
     - ANSI colour code for failures.  Default ``31`` (red).
   * - ``_success``
     - ``int``
     - ANSI colour code for success messages.  Default ``32`` (green).
   * - ``_info``
     - ``int``
     - ANSI colour code for info messages.  Default ``37`` (white).

Logging Methods
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Signature
     - Description
   * - ``void success(std::string message)``
     - Prints *message* in green (ANSI 32) unless ``shush`` is set.
   * - ``void warning(std::string message)``
     - Prints *message* in yellow (ANSI 33) unless ``shush`` is set.
   * - ``void failure(std::string message)``
     - Prints *message* in red (ANSI 31) unless ``shush`` is set.
   * - ``void info(std::string message)``
     - Prints *message* in white (ANSI 37) unless ``shush`` is set.

Progress-Bar Methods
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Signature
     - Description
   * - ``void progressbar(float prog, std::string title)``
     - Renders a single-bar terminal progress indicator at fraction *prog* (0–1).
   * - ``void progressbar(std::vector<size_t>* threads, std::vector<size_t>* trgt, std::vector<std::string>* title)``
     - Multi-bar variant: one progress bar per entry in *threads*, showing
       ``threads[i] / trgt[i]`` with label ``title[i]``.
   * - ``int running(std::vector<std::thread*>* thr, std::vector<size_t>* prg, std::vector<size_t>* trgt)``
     - Blocks until all threads in *thr* complete, rendering live progress bars.
       Returns the number of threads that finished without error.
   * - ``void monitor(std::vector<std::thread*>* thr)``
     - Joins all threads in *thr* without a progress bar.
   * - ``static void progressbar1(std::vector<size_t>* threads, size_t l, std::string title)``
     - Static single-bar helper used by worker threads.
   * - ``static void progressbar2(std::vector<size_t>* threads, size_t* l, std::string* title)``
     - Static multi-bar helper (pointer-based) for worker threads.
   * - ``static void progressbar3(std::vector<size_t>* threads, std::vector<size_t>* l, std::vector<std::string*>* title)``
     - Static multi-bar helper for a vector of progress targets.


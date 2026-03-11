Tools (Python)
==============

The ``Tools`` Cython class exposes the C++ ``tools`` utility functions.

Methods
-------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Signature
     - Description
   * - ``create_path(pth: str)``
     - Create the directory tree at *pth* (equivalent to ``mkdir -p``).
   * - ``delete_path(pth: str)``
     - Recursively delete the directory at *pth*.
   * - ``is_file(pth: str) → bool``
     - Return ``True`` if *pth* is a regular file.
   * - ``rename(src: str, dst: str)``
     - Rename/move file *src* to *dst*.
   * - ``abs(pth: str) → str``
     - Return the absolute path of *pth*.
   * - ``ls(pth: str, ext: str) → list[str]``
     - List files in directory *pth* with extension *ext*.
   * - ``replace(val: str, rpl: str, rpwl: str) → str``
     - Replace all occurrences of *rpl* in *val* with *rpwl*.
   * - ``has_substring(val: str, rpl: str) → bool``
     - Return ``True`` if *val* contains the substring *rpl*.
   * - ``ends_with(val: str, rpl: str) → bool``
     - Return ``True`` if *val* ends with the suffix *rpl*.
   * - ``has_value(data: list, trg: str) → bool``
     - Return ``True`` if the string list *data* contains *trg*.
   * - ``split(data: str, trg) → list[str]``
     - Split *data* on the delimiter(s) in *trg*.
   * - ``hash(data: str, lx: int = 8) → str``
     - Compute a MD5-based hex hash of *data*.  The result is always at
       least *lx* characters long; if the natural hash is shorter, zeros
       are appended.  ``lx=0`` gives the raw 36-character UUID-style hash.
   * - ``encode64(data: str) → str``
     - Base-64 encode *data*.

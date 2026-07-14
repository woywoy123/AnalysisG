Tools Module
============

The ``tools`` class is the primary utility base class, providing static helpers
for file-system operations, string manipulation, hashing, base-64 encoding, and
generic container algorithms. Most framework classes inherit ``tools`` so that
these helpers are always available via ``this->``.

Class: ``tools``
----------------

**Header:** ``<tools/tools.h>``

**Inheritance:** (base class — no parents)

File-System Methods
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Signature
     - Description
   * - ``static void create_path(std::string path)``
     - Recursively creates the given directory path (``mkdir -p`` equivalent).
   * - ``static void delete_path(std::string path)``
     - Deletes the file or directory at *path* (calls ``std::filesystem::remove_all``).
   * - ``static bool is_file(std::string path)``
     - Returns ``true`` if *path* points to a regular file.
   * - ``static void rename(std::string start, std::string target)``
     - Renames / moves *start* to *target*.
   * - ``static std::string absolute_path(std::string path)``
     - Returns the canonical absolute path of *path*.
   * - ``static std::vector<std::string> ls(std::string path, std::string ext = "")``
     - Lists all files under *path* whose extension matches *ext* (empty = all files).

String Methods
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Signature
     - Description
   * - ``static std::string to_string(double val)``
     - Converts a double to its string representation.
   * - ``static std::string to_string(double val, int prec)``
     - Converts a double to a string with *prec* significant digits.
   * - ``static void replace(std::string* in, std::string repl_str, std::string repl_with)``
     - Replaces the first occurrence of *repl_str* in ``*in`` with *repl_with* (in-place).
   * - ``static bool has_string(std::string* inpt, std::string trg)``
     - Returns ``true`` if ``*inpt`` contains the substring *trg*.
   * - ``static bool ends_with(std::string* inpt, std::string val)``
     - Returns ``true`` if ``*inpt`` ends with the suffix *val*.
   * - ``static bool has_value(std::vector<std::string>* data, std::string trg)``
     - Returns ``true`` if the vector *data* contains the string *trg*.
   * - ``static std::vector<std::string> split(std::string in, std::string delim)``
     - Splits *in* on the delimiter *delim* and returns the tokens.
   * - ``static std::vector<std::string> split(std::string in, size_t n)``
     - Splits *in* into substrings of length *n*.
   * - ``static std::string lower(std::string*)``
     - Returns a lower-cased copy of the input string.

Hashing and Encoding
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Signature
     - Description
   * - ``static std::string hash(std::string input, int len = 18)``
     - Returns a hex string of length *len* derived from *input* via SHA-256.
       The default ``len=18`` produces the 18-character hashes used as particle
       and event identifiers throughout the framework.
   * - ``static std::string encode64(std::string* data)``
     - Base-64 encodes ``*data`` and returns the encoded string.
   * - ``static std::string encode64(unsigned char const*, unsigned int len)``
     - Base-64 encodes a raw byte buffer of *len* bytes.
   * - ``static std::string decode64(std::string* inpt)``
     - Base-64 decodes ``*inpt`` and returns the decoded string.
   * - ``static std::string decode64(std::string const& s)``
     - Base-64 decodes the const-reference string *s*.

Template Utilities
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Signature
     - Description
   * - ``template<G> static std::vector<std::vector<G>> discretize(std::vector<G>* v, int N)``
     - Splits *v* into consecutive chunks of at most *N* elements.
   * - ``template<g> static g max(std::vector<g>* inpt)``
     - Returns the maximum value in *inpt*.
   * - ``template<g> static g min(std::vector<g>* inpt)``
     - Returns the minimum value in *inpt*.
   * - ``template<g> static g sum(std::vector<g>* inpt)``
     - Accumulates and returns the sum of all elements in *inpt*.
   * - ``template<g> static std::vector<g*> put(std::vector<g*>* src, std::vector<int>* trg)``
     - Returns a new vector by gathering elements of *src* at the indices *trg*.
   * - ``template<g> static void put(std::vector<g*>* out, std::vector<g*>* src, std::vector<int>* trg)``
     - In-place version: clears *out* and fills it by gathering from *src* at *trg*.
       Also sets ``in_use = 1`` on each selected element.
   * - ``template<g> static void unique_key(std::vector<g>* inx, std::vector<g>* oth)``
     - Appends elements of *inx* that are not already in *oth* to *oth*
       (union deduplication by value).


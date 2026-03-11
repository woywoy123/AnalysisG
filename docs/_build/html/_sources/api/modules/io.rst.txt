I/O Module
==========

The ``io`` class handles all reading and writing of physics data, supporting
both CERN ROOT files (via TFile/TTree) and HDF5 files (via H5Cpp).  It inherits
from ``tools`` and ``notification`` and exposes a template-based read/write API
for arbitrary C++ types.

Class: ``io``
--------------

**Header:** ``<io/io.h>``

**Inheritance:** ``tools``, ``notification``

HDF5 Methods
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Signature
     - Description
   * - ``bool start(std::string filename, std::string read_write)``
     - Opens (or creates) an HDF5 file at *filename*.
       *read_write* is ``"r"`` (read), ``"w"`` (write), or ``"rw"`` (read-write).
       Returns ``false`` on failure.
   * - ``void end()``
     - Closes the currently open HDF5 file.
   * - ``template<g> void write(std::vector<g>* inpt, std::string set_name)``
     - Writes the vector *inpt* to HDF5 dataset *set_name*.
       *g* must be a trivially copyable POD type.
   * - ``template<g> void write(g* inpt, std::string set_name)``
     - Writes a single scalar *inpt* to HDF5 dataset *set_name*.
   * - ``template<g> void read(std::vector<g>* outpt, std::string set_name)``
     - Reads HDF5 dataset *set_name* into *outpt* (resized automatically).
   * - ``template<g> void read(g* out, std::string set_name)``
     - Reads a single scalar from HDF5 dataset *set_name* into *out*.
   * - ``void read(graph_hdf5_w* out, std::string set_name)``
     - Reads a ``graph_hdf5_w`` struct from HDF5 dataset *set_name*.
   * - ``std::vector<std::string> dataset_names()``
     - Returns a list of all dataset names in the open HDF5 file.

ROOT Methods
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Signature
     - Description
   * - ``bool scan_keys()``
     - Scans all open ROOT files and builds the ``keys`` map.
       Must be called after setting ``trees``, ``branches``, and ``leaves``.
       Returns ``false`` if no keys were found.
   * - ``void root_begin()``
     - Begins an iteration over the ``keys`` map, setting up ``TTreeReader``
       instances.
   * - ``void root_end()``
     - Closes all open TTreeReader instances.
   * - ``void trigger_pcm()``
     - Triggers the pre-compiled macro (PCM) for object dictionaries.
   * - ``void check_root_file_paths()``
     - Validates that all file paths in ``root_files`` exist.
   * - ``std::map<std::string, long> root_size()``
     - Returns a map of tree-path → number of entries.
   * - ``std::map<std::string, data_t*>* get_data()``
     - Returns a pointer to the internal ``data_t`` map (populated by
       ``root_begin`` / ``root_end``).
   * - ``void import_settings(settings_t* params)``
     - Copies ``trees``, ``branches``, ``leaves``, and other settings
       from a ``settings_t`` struct.

Configuration Fields
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Field
     - Type
     - Description
   * - ``enable_pyami``
     - ``bool``
     - Enable AMI metadata fetching.  Default ``true``.
   * - ``metacache_path``
     - ``std::string``
     - Path for the local metadata JSON cache.  Default ``"./"``.
   * - ``current_working_path``
     - ``std::string``
     - Working directory used for relative file paths.  Default ``"."``.
   * - ``sow_name``
     - ``std::string``
     - Name of the sum-of-weights branch in the ROOT file.  Default ``""``.
   * - ``trees``
     - ``std::vector<std::string>``
     - List of ROOT tree names to scan (e.g. ``{"nominal"}``).
   * - ``branches``
     - ``std::vector<std::string>``
     - List of branch-path filter prefixes.
   * - ``leaves``
     - ``std::vector<std::string>``
     - List of leaf-path filter suffixes.

Internal ROOT Maps (populated by ``scan_keys``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Field
     - Description
   * - ``files_open``
     - Map of filename → open ``TFile*``.
   * - ``meta_data``
     - Map of filename → ``meta*`` objects.
   * - ``tree_data``
     - Map of filename → {tree-name → ``TTree*``}.
   * - ``tree_entries``
     - Map of filename → {tree-name → entry count}.
   * - ``branch_data``
     - Map of path → {branch-name → ``TBranch*``}.
   * - ``leaf_data``
     - Map of filename → {leaf-name → ``TLeaf*``}.
   * - ``leaf_typed``
     - Map of filename → {leaf-name → ROOT type string}.
   * - ``keys``
     - Full nested map: filename → tree → branch → list of leaf names.

Example::

    io reader;
    reader.trees = {"nominal"};
    reader.leaves = {"jet_pt"};
    reader.root_files["sample.root"] = true;
    reader.scan_keys();
    reader.root_begin();
    auto* data = reader.get_data();
    // iterate entries via data map
    reader.root_end();

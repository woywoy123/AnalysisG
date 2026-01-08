.. cpp:class:: io : public tools, public notification

    Handles input and output operations, primarily focused on interacting with HDF5 files for structured data storage and ROOT files for high-energy physics data analysis.

    This class provides functionalities to:

    *   Read and write C++ structs and vectors of structs to HDF5 datasets.
    *   Manage the lifecycle of HDF5 files (opening, closing).
    *   Discover dataset names within an HDF5 file.
    *   Scan specified ROOT files or directories containing ROOT files.
    *   Identify and access requested TTrees, TBranches, and TLeaves within ROOT files.
    *   Handle potential wildcards in ROOT file paths.
    *   Scrape metadata (like Sum of Weights) from ROOT files using associated ``meta`` objects.
    *   Set up iterators (``data_t``) for efficient reading of data from specified leaves across multiple ROOT files.
    *   Manage the lifecycle of opened ROOT files and associated iterators.
    *   Trigger the generation of ROOT Persistent Class Dictionaries (PCMs) if necessary.
    *   Import configuration settings.

    It inherits from ``tools`` (presumably providing utility functions like path manipulation, string operations) and ``notification`` (likely for logging messages like info, warning, success, failure).

    .. cpp:function:: io()

        Default constructor for the io class.

        Initializes the notification prefix to "io". Sets up the basic state of the io object, ready for configuration and file operations.

    .. cpp:function:: ~io()

        Destructor for the io class.

        Ensures proper cleanup of all managed resources. This includes:

        *   Closing any open HDF5 file via :cpp:func:`end()`.
        *   Cleaning up ROOT file reading resources (iterators, shared vectors) via :cpp:func:`root_end()`.
        *   Closing and deleting any TFile objects stored in the ``files_open`` map.
        *   Deleting any ``meta`` objects stored in the ``meta_data`` map.

        This prevents resource leaks (file handles, memory).

    .. cpp:function:: template <typename g> void write(std::vector<g>* inpt, std::string set_name)

        Writes a vector of objects to a named HDF5 dataset.

        :tparam g: The type of the objects contained within the vector. This type must have a corresponding private ``member(g)`` function defined within the ``io`` class to describe its structure as an HDF5 compound datatype.
        :param inpt: Pointer to the ``std::vector<g>`` containing the data to be written. The vector's size determines the dataset's dimension.
        :param set_name: The name to assign to the dataset within the HDF5 file. If a dataset with this name already exists, it might be overwritten depending on how the file was opened and the dataset creation properties.

        This function template serializes the contents of the input vector into an HDF5 dataset.

        #. It determines the HDF5 compound datatype corresponding to type ``g`` by calling ``member(g())``.
        #. It retrieves or creates an HDF5 dataset handle using ``dataset(set_name, type, length)``. The length is obtained from the input vector's size.
        #. If the dataset handle is valid (i.e., the file is open and the dataset could be created/accessed), it writes the entire vector's data (``inpt->data()``) to the dataset using the determined HDF5 type.
        #. It releases the HDF5 datatype resource (``H5Tclose``).

        If the dataset cannot be accessed or created (e.g., file not open), the function returns without writing.

    .. cpp:function:: template <typename g> void write(g* inpt, std::string set_name)

        Writes a single object to a named HDF5 dataset.

        :tparam g: The type of the object to be written. This type must have a corresponding private ``member(g)`` function defined within the ``io`` class to describe its structure as an HDF5 compound datatype.
        :param inpt: Pointer to the object of type ``g`` containing the data to be written.
        :param set_name: The name to assign to the dataset within the HDF5 file. The dataset will be created with a length of 1. If a dataset with this name already exists, it might be overwritten.

        This function template serializes a single C++ object into an HDF5 dataset.

        #. It determines the HDF5 compound datatype for type ``g`` via ``member(g())``.
        #. It retrieves or creates an HDF5 dataset handle using ``dataset(set_name, type, 1)``, specifying a length of 1.
        #. If the dataset handle is valid, it writes the data from the input object (``inpt``) to the dataset using the determined HDF5 type.
        #. It releases the HDF5 datatype resource (``H5Tclose``).

        If the dataset cannot be accessed or created, the function returns without writing.

    .. cpp:function:: template <typename g> void read(std::vector<g>* outpt, std::string set_name)

        Reads data from a named HDF5 dataset into a vector of objects.

        :tparam g: The type of the objects to be read into the vector. This type must have a corresponding private ``member(g)`` function defined within the ``io`` class to describe its HDF5 compound datatype structure, matching the structure of the data stored in the dataset.
        :param outpt: Pointer to the ``std::vector<g>`` where the read data will be stored. The vector will be cleared and resized to match the dimensions of the HDF5 dataset before reading.
        :param set_name: The name of the dataset to read from within the currently open HDF5 file.

        This function template deserializes data from an HDF5 dataset into a C++ vector.

        #. It determines the HDF5 compound datatype for type ``g`` via ``member(g())``.
        #. It retrieves an HDF5 dataset handle for reading using ``dataset(set_name)``.
        #. If the dataset handle is valid:
            a. It gets the dataspace of the dataset to determine its dimensions.
            b. It extracts the size (length) of the 1D dataset.
            c. It resizes the output vector (``outpt``) to match the dataset length, filling it with default-constructed objects of type ``g``.
            d. It reads the data from the HDF5 dataset into the vector's data buffer (``outpt->data()``) using the determined HDF5 type.
            e. It releases the HDF5 dataspace resource (``space_r.close()``).
        #. It releases the HDF5 datatype resource (``H5Tclose``).

        If the dataset cannot be accessed (e.g., file not open, dataset doesn't exist), the function returns without modifying the output vector.

    .. cpp:function:: template <typename g> void read(g* out, std::string set_name)

        Reads data from a named HDF5 dataset into a single object.

        :tparam g: The type of the object to read data into. This type must have a corresponding private ``member(g)`` function defined within the ``io`` class to describe its HDF5 compound datatype structure, matching the structure of the data stored in the dataset.
        :param out: Pointer to the object of type ``g`` where the read data will be stored. It is assumed the dataset contains data for at least one object; this function reads the first element from the dataset into the object.
        :param set_name: The name of the dataset to read from within the currently open HDF5 file.

        This function template deserializes data from an HDF5 dataset into a single C++ object.

        #. It determines the HDF5 compound datatype for type ``g`` via ``member(g())``.
        #. It retrieves an HDF5 dataset handle for reading using ``dataset(set_name)``.
        #. If the dataset handle is valid:
            a. It gets the dataspace (primarily to ensure the dataset exists, though dimensions aren't explicitly used here).
            b. It reads data from the beginning of the HDF5 dataset directly into the memory location pointed to by ``out``, using the determined HDF5 type.
            c. It releases the HDF5 dataspace resource (``space_r.close()``).
        #. It releases the HDF5 datatype resource (``H5Tclose``).

        If the dataset cannot be accessed, the function returns without modifying the output object.

        .. note:: This function reads only the first entry from the dataset, even if it contains multiple entries.

    .. cpp:function:: void read(graph_hdf5_w* out, std::string set_name)

        Reads data from an HDF5 dataset specifically into a graph_hdf5_w object.

        :param out: Pointer to the ``graph_hdf5_w`` object where the read data will be stored.
        :param set_name: The name of the dataset to read from within the HDF5 file. The dataset is expected to contain data structured according to the ``graph_hdf5_w`` layout.

        This is an explicit overload (or specialization pattern implemented via overload) for reading data structured as ``graph_hdf5_w``.

        #. It obtains the specific HDF5 compound datatype for ``graph_hdf5_w`` by calling ``member(graph_hdf5_w())``.
        #. It retrieves an HDF5 dataset handle for reading using ``dataset(set_name)``.
        #. If the dataset handle is valid:
            a. It gets the dataspace.
            b. It reads the first entry from the dataset into the provided ``out`` object.
            c. It releases the HDF5 dataspace resource.
        #. It releases the HDF5 datatype resource.

        If the dataset cannot be accessed, the function returns without modification.

        See also: ``member(graph_hdf5_w)``

    .. cpp:function:: bool start(std::string filename, std::string read_write)

        Opens or creates an HDF5 file for subsequent read/write operations.

        :param filename: The path (absolute or relative) to the HDF5 file. If the path doesn't exist and the mode is "write", the necessary directories will be created using ``create_path``.
        :param read_write: A string specifying the access mode:
                           - ``"write"``: Open for writing. If the file exists, it's opened in read-write mode (``H5F_ACC_RDWR``). If it doesn't exist, it's created, truncating any existing file with the same name (``H5F_ACC_TRUNC``).
                           - ``"read"``: Open for reading only (``H5F_ACC_RDONLY``). The file must already exist.
        :return: ``true`` if the file was successfully opened or created in the specified mode. ``false`` if an error occurred (e.g., trying to read a non-existent file, invalid ``read_write`` mode specified, or if another HDF5 file is already managed by this ``io`` instance (``this->file`` is not null)).

        This function manages the HDF5 file handle (``this->file``). It checks if a file is already open. If not, it determines the correct HDF5 access flag based on the ``read_write`` mode and file existence. It attempts to create parent directories if needed for writing. Finally, it creates a new ``H5::H5File`` object and stores the pointer in ``this->file``. Only one HDF5 file can be actively managed by an ``io`` instance at a time.

        See also: :cpp:func:`end()`, ``is_file()``, ``create_path()``

    .. cpp:function:: void end()

        Closes the currently open HDF5 file and releases associated resources.

        This function performs the necessary cleanup for HDF5 operations.

        #. It checks if an HDF5 file is currently open (``this->file`` is not null).
        #. If a file is open, it calls ``this->file->close()`` to close the HDF5 file handle.
        #. It deletes the ``H5::H5File`` object pointed to by ``this->file`` to free memory.
        #. It sets ``this->file`` back to ``nullptr``.
        #. It iterates through the cached HDF5 dataset handles stored in ``data_w`` (for writing) and ``data_r`` (for reading). For each cached handle:
            a. It calls ``close()`` on the ``H5::DataSet`` object.
            b. It deletes the ``H5::DataSet`` object.
        #. It clears both the ``data_w`` and ``data_r`` maps.

        This ensures that the HDF5 file is properly closed and all associated HDF5 objects (file, datasets) managed by this class are released.

        See also: :cpp:func:`start()`, ``dataset(std::string set_name, hid_t type, long long unsigned int length)``, ``dataset(std::string set_name)``

    .. cpp:function:: std::vector<std::string> dataset_names()

        Retrieves a list of all dataset names at the root level of the currently open HDF5 file.

        :return: ``std::vector<std::string>`` A vector containing the names of all objects identified as datasets within the root group of the HDF5 file. Returns an empty vector if no HDF5 file is currently open (``this->file`` is null).

        This function uses the HDF5 C API function ``H5Literate`` to iterate through all objects directly within the root group of the opened file (``this->file->getId()``). It uses the static callback function ``file_info`` for the iteration. The ``file_info`` function attempts to open each object as a dataset (``H5Dopen2``). If successful, it adds the object's name to the vector passed as ``opdata``. The resulting vector of dataset names is then returned.

        See also: ``file_info()``, :cpp:func:`start()`

    .. cpp:function:: std::map<std::string, long> root_size()

        Calculates the total number of entries for each requested TTree across all scanned ROOT files.

        :return: ``std::map<std::string, long>`` A map where keys are the names of the TTrees requested by the user (stored in ``this->trees``) and found during the scan. The values are the sum of entries for that specific TTree across all ROOT files where it was successfully located.

        This function first ensures that the ROOT file scanning process has been completed by calling :cpp:func:`scan_keys()` if it hasn't been run already (or if the results might be stale). It then iterates through the ``tree_entries`` map, which was populated by :cpp:func:`scan_keys()`. ``tree_entries`` maps file paths to another map containing tree names and their entry counts for that specific file (``map<file, map<tree, entries>>``). The function aggregates these counts, summing the entries for each unique tree name across all files, and stores the totals in the output map.

        See also: :cpp:func:`scan_keys()`, :cpp:member:`tree_entries`, :cpp:member:`trees`

    .. cpp:function:: void check_root_file_paths()

        Validates, expands, and canonicalizes the list of input ROOT file paths.

        This function processes the initial list of paths provided in the ``root_files`` map. It iterates through the input paths (keys of ``root_files``) and performs the following:

        #. **Wildcard Expansion:** If a path ends with ``*``, it treats the part before the ``*`` as a directory path. It uses ``ls()`` (from the ``tools`` base class) to find all files ending with ``.root`` within that directory and adds their absolute paths to a temporary map.
        #. **Directory Scanning:** If a path does not end with ``.root`` (implying it might be a directory), it uses ``ls()`` to find all ``.root`` files within that directory and adds their absolute paths to the temporary map.
        #. **File Validation:** If a path seems to point to a specific file (ends with ``.root``), it checks if the file actually exists using ``is_file()``. If it exists, its absolute path (obtained via ``absolute_path()``) is added to the temporary map.
        #. **Logging:** It logs success messages for each valid file/path found and warning messages for files specified but not found.

        Finally, it replaces the original ``root_files`` map with the temporary map containing only the validated, absolute paths to the ROOT files that will be processed.

        See also: :cpp:member:`root_files`, ``ls()`` (from tools base class), ``is_file()`` (from tools base class), ``absolute_path()`` (from tools base class), ``ends_with()`` (from tools base class)

    .. cpp:function:: bool scan_keys()

        Scans the validated ROOT files to locate requested TTrees, TBranches, and TLeaves.

        :return: ``true`` Currently, this function always returns true. Consider changing the return type to ``bool`` or ``void`` if specific error conditions should halt the process or be signaled differently.

        This function orchestrates the scanning of ROOT file contents.

        #. It iterates through the validated file paths stored in ``root_files`` (populated by :cpp:func:`check_root_file_paths`).
        #. For each file path:
            a. It obtains a ``TFile`` pointer. If the file is already open and cached in ``files_open``, it reuses the handle; otherwise, it opens the file in "READ" mode and caches it. It handles potential "zombie" files (files that couldn't be opened correctly).
            b. It calls the recursive helper function ``root_key_paths("")`` to traverse the directory structure within the ROOT file, starting from the root directory. This helper function populates ``tree_data``, ``tree_entries``, ``branch_data``, ``leaf_data``, ``leaf_typed``, and potentially interacts with ``meta_data``.
            c. It closes the ``TFile`` (the handle remains cached in ``files_open``).
        #. After scanning all files, it analyzes the results to determine which requested items (from ``trees``, ``branches``, ``leaves``) were found in which files.
        #. It populates the ``keys`` map, recording any missing items for each file.
        #. It logs detailed messages:
            - Success messages for files where all requested items were found, including the number of events per found tree (logged only once per file).
            - Failure messages for files where items were missing.
            - Warning messages listing the specific missing trees, branches, or leaves (logged only once per unique missing item name across all files, using ``missing_trigger``).

        See also: :cpp:func:`check_root_file_paths()`, ``root_key_paths(std::string path)``, :cpp:member:`files_open`, :cpp:member:`tree_data`, :cpp:member:`tree_entries`, :cpp:member:`branch_data`, :cpp:member:`leaf_data`, :cpp:member:`leaf_typed`, :cpp:member:`keys`, :cpp:member:`trees`, :cpp:member:`branches`, :cpp:member:`leaves`, ``missing_trigger``, ``success_trigger``

    .. cpp:function:: void root_begin()

        Initializes the ROOT data reading process by setting up ``data_t`` iterators for each requested leaf.

        This function prepares the system for efficiently reading data column-wise (leaf-wise) across multiple ROOT files.

        #. It ensures the file scanning is complete by calling :cpp:func:`scan_keys()`.
        #. If iterators (``iters``) already exist from a previous call, it cleans them up using :cpp:func:`root_end()`.
        #. It creates the main map (``iters``) to store pointers to ``data_t`` objects, keyed by the full leaf path.
        #. It creates a shared vector (``vx``) to hold ``TFile*`` pointers for all relevant files.
        #. It iterates through the ``leaf_data`` map (populated by :cpp:func:`scan_keys()`), which contains information about found leaves (``map<file, map<leaf_path, TLeaf*>>``).
        #. For each found leaf in each file:
            a. It checks if the corresponding tree has entries (skips leaves in empty trees).
            b. If an iterator (``data_t``) for this specific leaf path (``lf_name``, e.g., "Tree.Branch.Leaf") doesn't exist yet in the ``handl`` temporary storage:
                i. It creates a new ``data_t`` object.
                ii. It populates the ``data_t`` object with the leaf path, tree name, leaf name, and leaf type.
                iii. It creates shared vectors (``files_s`` for filenames, ``files_i`` for entry counts per file) and assigns the shared ``TFile*`` vector (``files_t = vx``).
                iv. It stores the new ``data_t`` pointer in ``handl`` and marks the leaf path in ``iters``.
            c. It retrieves the ``data_t`` object for the current leaf path from ``handl``.
            d. It adds the current file's name (``fname``) to the ``data_t``'s ``files_s`` vector.
            e. It adds the entry count for the corresponding tree in this file to the ``data_t``'s ``files_i`` vector.
            f. It ensures the ``TFile`` pointer for the current file is present in the shared ``files_t`` vector (``vx``), opening the file if necessary (only the first file needs explicit opening here, others reuse handles).
            g. If the ``data_t`` object hasn't been fully initialized yet (e.g., TTreeReader setup), it calls ``initialize()`` on it.
        #. Finally, it populates the main ``iters`` map by transferring the pointers from ``handl``.

        See also: :cpp:func:`scan_keys()`, :cpp:func:`root_end()`, :cpp:func:`get_data()`, :cpp:member:`iters`, :cpp:member:`leaf_data`, :cpp:member:`tree_entries`, :cpp:member:`files_open`, ``data_t::initialize()``, ``data_t``

    .. cpp:function:: void root_end()

        Cleans up resources associated with ROOT file reading iterators.

        This function releases the resources allocated and managed by :cpp:func:`root_begin`.

        #. It checks if the iterator map (``iters``) exists (i.e., if :cpp:func:`root_begin` was called).
        #. If ``iters`` exists:
            a. It iterates through each key-value pair in the ``iters`` map.
            b. For each ``data_t*`` pointer (``it->second``):
                i. It calls ``flush()`` on the ``data_t`` object, which likely releases internal ROOT resources like ``TTreeReader`` and associated value/array readers.
                ii. It captures the pointer to the shared ``files_t`` vector (if not already captured).
                iii. It deletes the ``data_t`` object itself.
            c. After iterating through all ``data_t`` objects, it deletes the shared vector of ``TFile*`` pointers (``fx``) that was used by all ``data_t`` objects.
            d. It clears the ``iters`` map.
            e. It deletes the ``iters`` map object itself and sets the ``iters`` pointer to ``nullptr``.

        .. note::
            This function specifically cleans up the iterator structures created by :cpp:func:`root_begin`. It does *not* close the ``TFile`` objects themselves; those are managed separately (e.g., in the destructor or potentially by explicit calls elsewhere). Closing happens in the :cpp:func:`~io()` destructor.

        See also: :cpp:func:`root_begin()`, :cpp:func:`get_data()`, :cpp:member:`iters`, ``data_t::flush()``, :cpp:func:`~io()`

    .. cpp:function:: void trigger_pcm()

        Triggers the generation of ROOT Persistent Class Dictionaries (PCMs) if deemed necessary.

        This function attempts to ensure that necessary ROOT dictionaries, specifically for ``meta_t`` and ``weights_t`` (and potentially others via ``buildAll``), are generated and available as PCM files.

        #. It determines the target directory for PCM files (``dict_path`` + "pcm/").
        #. It creates this directory if it doesn't exist.
        #. It counts the number of existing ``.pcm`` files in the target directory.
        #. It compares the count to an expected number (derived from ``data_enum::undef``, seemingly a heuristic). If the count is lower than expected, it proceeds with generation.
        #. If generation is needed:
            a. It logs a message indicating that PCMs are being built.
            b. It sets ROOT's build directory (``gSystem->SetBuildDir``) to the PCM path.
            c. It changes the current working directory to the PCM path.
            d. It adds the PCM path to ROOT's dynamic library path.
            e. It sets the ACLiC compilation mode (e.g., optimized).
            f. It launches separate threads to execute external functions (``buildDict``, ``buildAll``) to generate dictionaries for specific classes (``meta_t``, ``weights_t``) using their header file paths and potentially a general build function. It waits for each thread to complete (``join``).
            g. It changes the working directory back to the original path.

        .. note::
            This function relies heavily on external context:
            - The ``dict_path`` variable (presumably defined elsewhere, possibly in ``io/cfg.h``).
            - The external functions ``buildDict`` and ``buildAll`` (likely defined elsewhere, responsible for invoking ROOT's dictionary generation mechanism like ``rootcling`` or ACLiC).
            - The ``data_enum`` enum (likely defined elsewhere).
            - Assumes necessary ROOT environment variables and tools are available.

        See also: ``create_path()``, ``ls()``, ``absolute_path()``

    .. cpp:function:: void import_settings(settings_t* params)

        Imports relevant configuration settings from a ``settings_t`` object.

        :param params: Pointer to a ``settings_t`` object containing various configuration parameters for the analysis or processing task.

        This function copies specific settings from the provided ``settings_t`` object into the member variables of the ``io`` class instance. This allows configuring the behavior of the ``io`` class based on external settings. The following settings are imported:

        - ``enable_pyami``: Copied from ``params->fetch_meta``. Controls whether metadata fetching (potentially using PyAMI) is enabled.
        - ``metacache_path``: Copied from ``params->metacache_path``. Specifies the path for caching metadata (likely an HDF5 file).
        - ``sow_name``: Copied from ``params->sow_name``. Specifies the name of the TTree or object within ROOT files that contains Sum of Weights information.

        If ``sow_name`` is provided (not empty), it logs an informational message indicating the name being checked for Sum of Weights.

        See also: ``settings_t``, :cpp:member:`enable_pyami`, :cpp:member:`metacache_path`, :cpp:member:`sow_name`

    .. cpp:function:: std::map<std::string, data_t*>* get_data()

        Provides access to the map of configured data iterators (``data_t``) for ROOT file reading.

        :return: ``std::map<std::string, data_t*>*`` Pointer to the internal map (``iters``) where keys are the full leaf paths (e.g., "TreeName.BranchName.LeafName") and values are pointers to the corresponding ``data_t`` iterator objects. Returns ``nullptr`` if :cpp:func:`root_begin()` has not been successfully called yet.

        This function serves as the primary way for external code to access the data iterators after they have been set up.

        #. It checks if the internal iterator map (``iters``) is currently null.
        #. If ``iters`` is null, it means :cpp:func:`root_begin()` has not been called or :cpp:func:`root_end()` was called since. In this case, it calls :cpp:func:`root_begin()` to initialize the iterators.
        #. It returns the pointer to the (potentially newly created) ``iters`` map.

        The caller can then use this map to retrieve specific ``data_t`` objects based on leaf names and use them to read data event by event.

        See also: :cpp:func:`root_begin()`, :cpp:func:`root_end()`, :cpp:member:`iters`, ``data_t``

    .. cpp:member:: bool enable_pyami

        Flag to control whether metadata fetching (e.g., via PyAMI or local cache) should be attempted. Defaults to ``true``.

    .. cpp:member:: std::string metacache_path

        Path to the file used for caching metadata (typically HDF5). Defaults to ``"./"``, which might be adjusted later (e.g., in ``root_key_paths``) to ``"./meta.h5"``.

    .. cpp:member:: std::string current_working_path

        Stores the current working directory, potentially used for resolving relative paths. Defaults to ``"."``.

    .. cpp:member:: std::string sow_name

        Name of the TTree or object expected to contain Sum of Weights information in ROOT files. Can be a simple name or include wildcards/delimiters for pattern matching (e.g., "pattern1*:pattern2"). Defaults to ``""``.

    .. cpp:member:: std::vector<std::string> trees

        Vector storing the names of TTrees the user wants to read data from. Populated externally before scanning.

    .. cpp:member:: std::vector<std::string> branches

        Vector storing the names of TBranches the user wants to read data from. Populated externally before scanning.

    .. cpp:member:: std::vector<std::string> leaves

        Vector storing the names of TLeaves the user wants to read data from. Populated externally before scanning.

    .. cpp:member:: std::map<std::string, TFile*> files_open

        Map caching pointers to opened ``TFile`` objects to avoid reopening. Key: absolute file path (``std::string``), Value: ``TFile*``. Populated during :cpp:func:`scan_keys`.

    .. cpp:member:: std::map<std::string, meta*> meta_data

        Map storing pointers to ``meta`` objects, one potentially associated with each ROOT file for metadata scraping. Key: absolute file path (``std::string``), Value: ``meta*``. Populated during ``root_key_paths``.

    .. cpp:member:: std::map<std::string, std::map<std::string, TTree*>> tree_data

        Map storing pointers to found ``TTree`` objects. Key: absolute file path (``std::string``), Value: ``map<tree_name, TTree*>``. Populated by ``root_key_paths``.

    .. cpp:member:: std::map<std::string, std::map<std::string, long>> tree_entries

        Map storing the number of entries for each found ``TTree``. Key: absolute file path (``std::string``), Value: ``map<tree_name, n_entries (long)>``. Populated by ``root_key_paths``.

    .. cpp:member:: std::map<std::string, std::map<std::string, TBranch*>> branch_data

        Map storing pointers to found ``TBranch`` objects. Key: absolute file path (``std::string``), Value: ``map<full_branch_path (Tree.Branch), TBranch*>``. Populated by ``root_key_paths``.

    .. cpp:member:: std::map<std::string, std::map<std::string, TLeaf*>> leaf_data

        Map storing pointers to found ``TLeaf`` objects. Key: absolute file path (``std::string``), Value: ``map<full_leaf_path (Tree.Branch.Leaf), TLeaf*>``. Populated by ``root_key_paths``.

    .. cpp:member:: std::map<std::string, std::map<std::string, std::string>> leaf_typed

        Map storing the ROOT data type name for each found ``TLeaf``. Key: absolute file path (``std::string``), Value: ``map<full_leaf_path, type_name (std::string)>``. Populated by ``root_key_paths``.

    .. cpp:member:: std::map<std::string, bool> root_files

        Map storing the validated and expanded list of absolute ROOT file paths to be processed. Key: absolute file path (``std::string``), Value: ``bool`` (typically ``true``). Populated/updated by :cpp:func:`check_root_file_paths`.

    .. cpp:member:: std::map<std::string, std::map<std::string, std::map<std::string, std::vector<std::string>>>> keys

        Map storing detailed information about missing requested items (Trees, Branches, Leaves) per file.
        Structure: ``map<file_path, map<"missed", map<key_type, vector<missing_names>>>>``

        - Outer Key: Absolute file path (``std::string``).
        - Middle Key: Always the string ``"missed"``.
        - Inner Key: Type of missing item (``"Trees"``, ``"Branches"``, or ``"Leaves"``).
        - Value: ``std::vector<std::string>`` containing the names of the missing items of that type for that file.

        Populated by :cpp:func:`scan_keys` after comparing requested items with found items.

    .. cpp:function:: hid_t member(folds_t t)
        :private:

        Creates and returns the HDF5 compound datatype definition for the ``folds_t`` struct.

        :param t: An instance of ``folds_t``. This parameter is only used for type deduction by the compiler and its value is ignored. It allows calling this function like ``member(folds_t())``.
        :return: ``hid_t`` The HDF5 datatype identifier (``hid_t``) representing the compound type corresponding to the ``folds_t`` struct. The caller is responsible for closing this identifier using ``H5Tclose()`` when it's no longer needed.

        This private helper function defines the memory layout and corresponding HDF5 types for each member of the ``folds_t`` struct. It uses ``H5Tcreate`` to create a new compound datatype and ``H5Tinsert`` to add each member:

        - ``k``: Mapped to ``H5T_NATIVE_INT``.
        - ``is_train``: Mapped to ``H5T_NATIVE_HBOOL``.
        - ``is_valid``: Mapped to ``H5T_NATIVE_HBOOL``.
        - ``is_eval``: Mapped to ``H5T_NATIVE_HBOOL``.
        - ``hash``: Mapped to a variable-length string type (``H5T_VARIABLE`` based on ``H5T_C_S1``).

        The offsets of members within the struct are calculated using the ``HOFFSET`` macro.

        See also: ``folds_t``, :cpp:func:`write()`, :cpp:func:`read()`

    .. cpp:function:: hid_t member(graph_hdf5_w t)
        :private:

        Creates and returns the HDF5 compound datatype definition for the ``graph_hdf5_w`` struct.

        :param t: An instance of ``graph_hdf5_w``. Used only for type deduction; its value is ignored. Allows calling like ``member(graph_hdf5_w())``.
        :return: ``hid_t`` The HDF5 datatype identifier (``hid_t``) for the compound type matching ``graph_hdf5_w``. The caller must close this identifier using ``H5Tclose()``.

        Defines the memory layout and HDF5 types for the ``graph_hdf5_w`` struct, used for writing graph data to HDF5.

        - ``num_nodes``: Mapped to ``H5T_NATIVE_INT``.
        - ``event_index``: Mapped to ``H5T_NATIVE_LONG``.
        - ``event_weight``: Mapped to ``H5T_NATIVE_DOUBLE``.
        - All ``std::string`` members (``hash``, ``filename``, ``edge_index``, ``data_map_*``, ``truth_map_*``, ``data_*``, ``truth_*``): Mapped to variable-length string types (``H5T_VARIABLE`` based on ``H5T_C_S1``).

        Uses ``H5Tcreate`` and ``H5Tinsert`` with ``HOFFSET`` for member definition.

        See also: ``graph_hdf5_w``, :cpp:func:`write()`, :cpp:func:`read(graph_hdf5_w* out, std::string set_name)`

    .. cpp:function:: static herr_t file_info(hid_t loc_id, const char* name, const H5L_info_t* linfo, void *opdata)
        :private:

        Callback function used by ``H5Literate`` to identify and collect dataset names.

        :param loc_id: The HDF5 identifier for the group being iterated (unused in this implementation).
        :param name: The name of the current object found by ``H5Literate``.
        :param linfo: Pointer to link information structure (unused in this implementation).
        :param opdata: A void pointer passed through ``H5Literate``. In this case, it's expected to be a ``reinterpret_cast``-ed pointer to a ``std::vector<std::string>``.
        :return: ``herr_t`` Returns 0 on success, indicating iteration should continue. A negative value would stop the iteration.

        This static function is designed to be used as the ``op`` argument in ``H5Literate``. For each object name passed by ``H5Literate``:

        #. It attempts to open the object using its ``name`` within the current location (``loc_id``) as an HDF5 dataset using ``H5Dopen2``.
        #. If ``H5Dopen2`` succeeds (meaning the object is a dataset), it casts ``opdata`` back to ``std::vector<std::string>*`` and pushes the ``name`` onto the vector.
        #. It closes the dataset handle obtained in step 1 using ``H5Dclose``.
        #. It returns 0 to continue the ``H5Literate`` process.

        If ``H5Dopen2`` fails (the object is not a dataset), the name is not added, and it still returns 0.

        See also: :cpp:func:`dataset_names()`, ``H5Literate`` (HDF5 C API documentation)

    .. cpp:member:: std::map<std::string, H5::DataSet*> data_w
        :private:

        Map caching opened HDF5 dataset handles (``H5::DataSet*``) intended for writing. Key: dataset name (``std::string``). Cleared by :cpp:func:`end()`.

    .. cpp:member:: std::map<std::string, H5::DataSet*> data_r
        :private:

        Map caching opened HDF5 dataset handles (``H5::DataSet*``) intended for reading. Key: dataset name (``std::string``). Cleared by :cpp:func:`end()`.

    .. cpp:member:: H5::H5File* file
        :private:

        Pointer to the currently managed ``H5::H5File`` object. ``nullptr`` if no HDF5 file is currently open via :cpp:func:`start()`. Managed by :cpp:func:`start()` and :cpp:func:`end()`.

    .. cpp:function:: H5::DataSet* dataset(std::string set_name, hid_t type, long long unsigned int length)
        :private:

        Gets or creates an HDF5 dataset handle for writing operations.

        :param set_name: The desired name of the dataset within the HDF5 file.
        :param type: The HDF5 datatype identifier (``hid_t``) describing the structure of the data to be written to the dataset (e.g., obtained from ``member()``).
        :param length: The desired number of elements (rows) for the dataset. This defines the first dimension of the 1D dataspace created for the dataset.
        :return: ``H5::DataSet*`` Pointer to the ``H5::DataSet`` object. Returns ``nullptr`` if the HDF5 file is not open (``this->file`` is null). The returned pointer points to an object managed by the ``data_w`` cache.

        This function provides access to HDF5 datasets for writing, using a cache (``data_w``) to avoid repeatedly creating/opening datasets.

        #. Checks if the main HDF5 file handle (``this->file``) is valid. If not, returns ``nullptr``.
        #. Checks if a dataset handle for ``set_name`` already exists in the ``data_w`` cache. If yes, returns the cached pointer.
        #. If not cached:
            a. Creates a 1-dimensional HDF5 dataspace (``H5::DataSpace``) with the specified ``length``.
            b. Suppresses HDF5 default error printing (``H5::Exception::dontPrint()``).
            c. Attempts to create the dataset within the HDF5 file (``this->file->createDataSet``) using the given ``set_name``, ``type``, and ``space``.
            d. Creates a new ``H5::DataSet`` object to wrap the HDF5 dataset ID.
            e. Stores the pointer to the new ``H5::DataSet`` object in the ``data_w`` cache, keyed by ``set_name``.
            f. Returns the pointer to the newly created and cached ``H5::DataSet`` object.

        HDF5 exceptions during dataset creation are caught implicitly by the HDF5 C++ wrappers but not explicitly handled here beyond suppression.

        See also: :cpp:func:`end()`, :cpp:func:`write()`, :cpp:member:`data_w`, :cpp:member:`file`

    .. cpp:function:: H5::DataSet* dataset(std::string set_name)
        :private:

        Gets an HDF5 dataset handle for reading operations.

        :param set_name: The name of the existing dataset to open within the HDF5 file.
        :return: ``H5::DataSet*`` Pointer to the ``H5::DataSet`` object. Returns ``nullptr`` if the HDF5 file is not open (``this->file`` is null). The returned pointer points to an object managed by the ``data_r`` cache.

        This function provides access to HDF5 datasets for reading, using a cache (``data_r``) to avoid repeatedly opening datasets.

        #. Checks if the main HDF5 file handle (``this->file``) is valid. If not, returns ``nullptr``.
        #. Checks if a dataset handle for ``set_name`` already exists in the ``data_r`` cache. If yes, returns the cached pointer.
        #. If not cached:
            a. Attempts to open the existing dataset within the HDF5 file (``this->file->openDataSet``) using the given ``set_name``.
            b. Creates a new ``H5::DataSet`` object to wrap the HDF5 dataset ID.
            c. Stores the pointer to the new ``H5::DataSet`` object in the ``data_r`` cache, keyed by ``set_name``.
            d. Returns the pointer to the newly opened and cached ``H5::DataSet`` object.

        Assumes the dataset named ``set_name`` exists in the file. Errors during opening (e.g., dataset not found) will likely result in exceptions from the HDF5 C++ library.

        See also: :cpp:func:`end()`, :cpp:func:`read()`, :cpp:member:`data_r`, :cpp:member:`file`

    .. cpp:member:: TFile* file_root
        :private:

        Pointer to the ``TFile`` object currently being scanned by ``root_key_paths``. Set during the loop in :cpp:func:`scan_keys`. ``nullptr`` otherwise.

    .. cpp:function:: void root_key_paths(std::string path)
        :private:

        Recursively scans directories and objects within the current ROOT file (``file_root``) to find TTrees, metadata, etc.

        :param path: The current path within the ROOT file's directory structure being scanned (e.g., "" for root, "subdir/" for a subdirectory).

        This is the main recursive function for exploring the content of a single ROOT file (``this->file_root``).

        #. Gets the current ``TDirectory`` (initially the file's root directory).
        #. Lists all keys (object names) within the current directory.
        #. Iterates through the keys:
            a. Retrieves the ``TObject`` corresponding to the key. Skips if the object cannot be retrieved.
            b. Checks if the object's name matches known metadata object names ("AnalysisTracking", "EventLoop_FileExecuted", "metadata", "MetaData") or matches the pattern(s) specified in ``sow_name``.
            c. If it's a potential metadata object:
                i. Ensures a ``meta`` object exists for the current file in ``meta_data``.
                ii. Configures the ``meta`` object with the ``metacache_path`` and filename.
                iii. Calls ``meta->scan_data(obj)`` to let the ``meta`` object extract relevant information.
            d. Checks if the object inherits from ``TTree``. If yes, calls the ``root_key_paths(path, TTree*)`` overload to process the tree.
            e. Checks if the object inherits from ``TH1``. If yes, skips it (histograms are ignored).
            f. Checks if the object is a ``TDirectory``. If yes, changes the current directory (``cd``) into the subdirectory, recursively calls ``root_key_paths`` with the updated path, and then changes back to the parent directory.

        This recursive traversal populates the ``tree_data``, ``tree_entries``, ``branch_data``, ``leaf_data``, ``leaf_typed``, and ``meta_data`` maps for the currently scanned file (``file_root``).

        See also: :cpp:func:`scan_keys()`, ``root_key_paths(std::string path, TTree* t)``, :cpp:member:`file_root`, :cpp:member:`meta_data`, :cpp:member:`sow_name`, :cpp:member:`metacache_path`, ``meta::scan_data()``

    .. cpp:function:: void root_key_paths(std::string path, TTree* t)
        :private:

        Scans a specific TTree for requested branches and leaves.

        :param path: The full path/name of the TTree object within the ROOT file (passed from the caller, potentially derived from the object name).
        :param t: Pointer to the ``TTree`` object to be scanned.

        This function is called by ``root_key_paths(std::string path)`` when a ``TTree`` object is encountered.

        #. Retrieves the ``TTree`` pointer again using ``file_root->Get<TTree>(path.c_str())`` (redundant if ``t`` is valid, maybe for safety). Skips if null.
        #. Gets the filename associated with ``file_root``.
        #. Checks if the name of the TTree ``t`` matches any of the names listed in ``this->trees``.
        #. If a match is found:
            a. Stores the ``TTree*`` pointer ``t`` in ``tree_data`` map, keyed by filename and tree name.
            b. Stores the number of entries (``t->GetEntries()``) in the ``tree_entries`` map, keyed similarly.
            c. Proceeds to scan for requested branches and leaves within this tree.
        #. If the tree name was not requested, the function returns early.
        #. **Branch Scanning:** Iterates through the requested branch names in ``this->branches``.
            a. For each requested name, calls ``t->FindBranch()`` to locate the ``TBranch``.
            b. If found, stores the ``TBranch*`` pointer in the ``branch_data`` map, keyed by filename and the fully qualified branch name (e.g., "TreeName.BranchName").
        #. **Leaf Scanning:** Iterates through the requested leaf names in ``this->leaves``.
            a. For each requested name, calls ``t->FindLeaf()`` to locate the ``TLeaf``.
            b. If found:
                i. Determines the full path of the leaf (e.g., "TreeName.LeafName" or "TreeName.BranchName.LeafName").
                ii. Stores the ``TLeaf*`` pointer in the ``leaf_data`` map, keyed by filename and full leaf path.
                iii. Stores the leaf's data type name (``lf->GetTypeName()``) in the ``leaf_typed`` map, keyed similarly.
                iv. Handles cases where the leaf belongs directly to the tree or is nested within a branch (including ``TBranchElement`` / folder branches by checking ``IsFolder`` and iterating sub-leaves if necessary, although the provided code snippet seems to handle nested leaves directly via ``FindLeaf`` and path construction).

        See also: ``root_key_paths(std::string path)``, :cpp:member:`trees`, :cpp:member:`branches`, :cpp:member:`leaves`, :cpp:member:`tree_data`, :cpp:member:`tree_entries`, :cpp:member:`branch_data`, :cpp:member:`leaf_data`, :cpp:member:`leaf_typed`, :cpp:member:`file_root`

    .. cpp:member:: std::map<std::string, data_t*>* iters
        :private:

        Pointer to the map holding the configured ``data_t`` iterators. Key: full leaf path (``std::string``), Value: ``data_t*``. Managed by :cpp:func:`root_begin()` and :cpp:func:`root_end()`. ``nullptr`` when not initialized.

    .. cpp:member:: std::map<std::string, bool> missing_trigger
        :private:

        Map used to track which missing item names (Trees, Branches, Leaves) have already been logged as warnings to avoid redundant messages. Key: item name (``std::string``), Value: ``bool`` (true if logged). Used in :cpp:func:`scan_keys`.

    .. cpp:member:: std::map<std::string, bool> success_trigger
        :private:

        Map used to track which files have already had a success message logged (indicating all requested items were found) to avoid redundant messages. Key: absolute file path (``std::string``), Value: ``bool`` (true if logged). Used in :cpp:func:`scan_keys`.

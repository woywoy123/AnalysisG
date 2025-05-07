/**
 * @file io.h
 * @brief Defines the `io` class for handling input/output operations, primarily with ROOT and HDF5 files.
 *
 * This file declares the `io` class, which encapsulates functionalities for reading and writing
 * data, particularly for physics analysis. It supports operations on ROOT TTrees and HDF5 datasets.
 * The class inherits from `tools` and `notification` for utility functions and logging.
 */

#ifndef IO_IO_H ///< Start of include guard for IO_IO_H to prevent multiple inclusions.
#define IO_IO_H ///< Definition of IO_IO_H to signify the header has been included.

#include <map> ///< Includes the standard map container.
#include <string> ///< Includes the standard string library.
#include <H5Cpp.h> ///< Includes the HDF5 C++ API header.

#include <TFile.h> ///< Includes ROOT's TFile class for file operations.
#include <TTree.h> ///< Includes ROOT's TTree class for tree-like data structures.
#include <TBranch.h> ///< Includes ROOT's TBranch class for TTree branches.
#include <TLeaf.h> ///< Includes ROOT's TLeaf class for TTree leaves.

#include <TTreeReader.h> ///< Includes ROOT's TTreeReader for efficient TTree reading.
#include <TTreeReaderArray.h> ///< Includes ROOT's TTreeReaderArray for reading array-like branches.

#include <meta/meta.h> ///< Includes the `meta` class, likely for metadata handling.
#include <tools/tools.h> ///< Includes the `tools` class for utility functions.
#include <structs/folds.h> ///< Includes `folds_t` structure, possibly for k-fold or data partitioning info.
#include <structs/element.h> ///< Includes `element_t` structure, a base data element type.
#include <structs/settings.h> ///< Includes `settings_t` structure for configuration settings.

#include <notification/notification.h> ///< Includes the `notification` class for logging and messages.

/**
 * @class io
 * @brief Manages input and output operations for analysis data.
 *
 * The `io` class provides an interface for reading from and writing to various data formats,
 * with a focus on ROOT files (TTrees) and HDF5 files. It handles file opening/closing,
 * data reading/writing for different data types (single objects, vectors of objects),
 * and manages internal data structures for tracking file and dataset information.
 */
class io: // Defines the class 'io'.
    public tools, ///< Inherits from the `tools` class for utility functions.
    public notification ///< Inherits from the `notification` class for logging capabilities.
{
public: ///< Public access specifier for the following members.
    /**
     * @brief Constructor for the `io` class.
     * Initializes a new io instance.
     */
    io(); 
    /**
     * @brief Destructor for the `io` class.
     * Cleans up resources, such as closing open files.
     */
    ~io(); 
   
    /**
     * @brief Templated method to write a vector of objects to a dataset.
     * @tparam g The type of objects in the vector.
     * @param inpt Pointer to a vector of objects of type `g` to be written.
     * @param set_name The name of the dataset to write to.
     */
    template <typename g>
    void write(std::vector<g>* inpt, std::string set_name){
        std::string type = this -> tools::type_name<g>(); ///< Get the type name of g.
        if (!this -> m_write){this -> Warning("NOT IN WRITE MODE"); return;} ///< Check if in write mode, issue warning and return if not.
        if (!this -> m_H5event){this -> write_event_h5(inpt, set_name); return;} ///< If HDF5 event writing is not active, call specific HDF5 event write method.
        if (this -> m_h5 -> findGroup(set_name)){this -> m_h5 -> rmGroup(set_name);} ///< If group exists in HDF5, remove it.
        this -> m_h5 -> template write_hdf5<g>(inpt, set_name); ///< Write data to HDF5.
    } 
 
    /**
     * @brief Templated method to write a single object to a dataset.
     * @tparam g The type of the object.
     * @param inpt Pointer to an object of type `g` to be written.
     * @param set_name The name of the dataset to write to.
     */
    template <typename g>
    void write(g* inpt, std::string set_name){
        std::string type = this -> tools::type_name<g>(); ///< Get the type name of g.
        if (!this -> m_write){this -> Warning("NOT IN WRITE MODE"); return;} ///< Check if in write mode, issue warning and return if not.
        if (!this -> m_H5event){this -> write_event_h5(inpt, set_name); return;} ///< If HDF5 event writing is not active, call specific HDF5 event write method.
        if (this -> m_h5 -> findGroup(set_name)){this -> m_h5 -> rmGroup(set_name);} ///< If group exists in HDF5, remove it.
        this -> m_h5 -> template write_hdf5<g>(inpt, set_name); ///< Write data to HDF5.
    }

    /**
     * @brief Templated method to read a vector of objects from a dataset.
     * @tparam g The type of objects in the vector.
     * @param outpt Pointer to a vector of objects of type `g` where read data will be stored.
     * @param set_name The name of the dataset to read from.
     */
    template <typename g>
    void read(std::vector<g>* outpt, std::string set_name){
        if (this -> m_write){this -> Warning("NOT IN READ MODE"); return;} ///< Check if in read mode, issue warning and return if not.
        if (!this -> m_H5event){this -> read_event_h5(outpt, set_name); return;} ///< If HDF5 event reading is not active, call specific HDF5 event read method.
        this -> m_h5 -> template read_hdf5<g>(outpt, set_name); ///< Read data from HDF5.
    } 

    /**
     * @brief Templated method to read a single object from a dataset.
     * @tparam g The type of the object.
     * @param out Pointer to an object of type `g` where read data will be stored.
     * @param set_name The name of the dataset to read from.
     */
    template <typename g>
    void read(g* out, std::string set_name){
        if (this -> m_write){this -> Warning("NOT IN READ MODE"); return;} ///< Check if in read mode, issue warning and return if not.
        if (!this -> m_H5event){this -> read_event_h5(out, set_name); return;} ///< If HDF5 event reading is not active, call specific HDF5 event read method.
        this -> m_h5 -> template read_hdf5<g>(out, set_name); ///< Read data from HDF5.
    }

    /**
     * @brief Reads graph data in HDF5 format specifically for `graph_hdf5_w` struct.
     * @param out Pointer to a `graph_hdf5_w` object to store the read data.
     * @param set_name The name of the dataset to read from.
     */
    void read(graph_hdf5_w* out, std::string set_name);

    /**
     * @brief Starts an I/O session, opening a file for reading or writing.
     * @param filename The name of the file to open.
     * @param read_write Mode of operation: "read" or "write".
     * @return True if the file was successfully opened in the specified mode, false otherwise.
     */
    bool start(std::string filename, std::string read_write); 
    /**
     * @brief Ends the current I/O session, closing any open files.
     */
    void end();
   
    /**
     * @brief Retrieves the names of datasets available in the currently open HDF5 file.
     * @return A vector of strings, where each string is a dataset name.
     */
    std::vector<std::string> dataset_names(); 

    /**
     * @brief Retrieves the sizes (number of entries) of TTrees in the open ROOT files.
     * @return A map where keys are TTree names (or unique identifiers) and values are their sizes (long).
     */
    std::map<std::string, long> root_size(); 
    /**
     * @brief Checks the validity or accessibility of ROOT file paths defined in settings or configurations.
     */
    void check_root_file_paths(); 
    /**
     * @brief Scans for keys (e.g., TTree names, TBranch names) in the open ROOT files.
     * @return True if keys were successfully scanned, false otherwise (e.g., no files open).
     */
    bool scan_keys(); 
    /**
     * @brief Initializes ROOT file processing, potentially opening files specified in settings.
     */
    void root_begin(); 
    /**
     * @brief Finalizes ROOT file processing, closing any ROOT files opened by `root_begin`.
     */
    void root_end(); 
    /**
     * @brief Triggers the generation of a ROOT PCM (Precompiled Module) if needed for dictionary generation.
     */
    void trigger_pcm(); 
    /**
     * @brief Imports settings from a `settings_t` object to configure the `io` instance.
     * @param params Pointer to a `settings_t` object.
     */
    void import_settings(settings_t* params); 

    /**
     * @brief Retrieves the data read from ROOT files, likely structured as a map of `data_t` objects.
     * @return Pointer to a map where keys are data identifiers (e.g., branch names) and values are `data_t` pointers.
     */
    std::map<std::string, data_t*>* get_data(); 

    bool enable_pyami = true; ///< Flag to enable or disable PyAMI (ATLAS Metadata Interface) usage. Default true.
    std::string metacache_path = "./"; ///< Path to the directory for caching metadata. Default current directory.
    std::string current_working_path = "."; ///< Current working directory path. Default ".".
    std::string sow_name = ""; ///< Name for Sum-Of-Weights information, if applicable. Default empty.

    std::vector<std::string> trees = {}; ///< Vector to store names of TTrees to be processed. Initialized empty.
    std::vector<std::string> branches = {}; ///< Vector to store names of TBranches to be processed. Initialized empty.
    std::vector<std::string> leaves = {}; ///< Vector to store names of TLeaves (variables) to be processed. Initialized empty.

    std::map<std::string, TFile*> files_open = {}; ///< Map storing pointers to open TFile objects, keyed by filename. Initialized empty.
    std::map<std::string, meta*>   meta_data = {}; ///< Map storing pointers to `meta` objects (metadata), keyed by an identifier (e.g., filename or sample label). Initialized empty.

    // key: Filename ; key tree_name : TTree*
    std::map<std::string, std::map<std::string, TTree*>> tree_data  = {}; ///< Nested map storing TTree pointers: Filename -> (Tree Name -> TTree*). Initialized empty.
    std::map<std::string, std::map<std::string, long>> tree_entries = {}; ///< Nested map storing TTree entry counts: Filename -> (Tree Name -> N_Entries). Initialized empty.
    // branch path : key branch_name : TBranch*
    std::map<std::string, std::map<std::string, TBranch*>> branch_data = {}; ///< Nested map storing TBranch pointers: Branch Path (e.g., File/Tree) -> (Branch Name -> TBranch*). Initialized empty.

    // leaf filename : key leaf_name : TLeaf*
    std::map<std::string, std::map<std::string, TLeaf*>>      leaf_data = {}; ///< Nested map storing TLeaf pointers: Filename -> (Leaf Name -> TLeaf*). Initialized empty.
    std::map<std::string, std::map<std::string, std::string>> leaf_typed = {}; ///< Nested map storing TLeaf type names: Filename -> (Leaf Name -> Type String). Initialized empty.
    std::map<std::string, bool> root_files = {}; ///< Map indicating if a given string (filename) corresponds to a (successfully opened) ROOT file. Initialized empty.

    std::map<std::string, std::map<std::string, std::map<std::string, std::vector<std::string>>>> keys; ///< Complex nested map for storing scanned keys from ROOT files (e.g., File -> Tree -> Branch -> Leaves).

private: ///< Private access specifier for the following members.
    /**
     * @brief HDF5 member type creation for `folds_t` struct.
     * @param t An instance of `folds_t` (used for type deduction, not value).
     * @return HDF5 data type identifier (`hid_t`).
     */
    hid_t member(folds_t t); 
    /**
     * @brief HDF5 member type creation for `graph_hdf5_w` struct.
     * @param t An instance of `graph_hdf5_w` (used for type deduction, not value).
     * @return HDF5 data type identifier (`hid_t`).
     */
    hid_t member(graph_hdf5_w t); 

    /**
     * @brief Static callback function for H5Literate, used to iterate over objects in an HDF5 file.
     * @param loc_id Location identifier.
     * @param name Name of the object.
     * @param linfo Pointer to H5L_info_t struct containing link information.
     * @param opdata Pointer to operator data (user-supplied).
     * @return HDF5 error status.
     */
    static herr_t file_info(hid_t loc_id, const char* name, const H5L_info_t* linfo, void *opdata); 
    /**
     * @brief Static callback function for H5Literate (alternative version or specific use case).
     * @param loc_id Location identifier.
     * @param name Name of the object.
     * @param linfo Pointer to H5L_info_t struct containing link information.
     * @param opdata Pointer to operator data (user-supplied).
     * @return HDF5 error status.
     */
    static herr_t op_func (hid_t loc_id, const char *name, const H5L_info_t *linfo, void *opdata); 

    /**
     * @brief Templated method to write event data (vector of objects) to HDF5.
     * Specific implementation for event-like data structures.
     * @tparam g Type of objects in the vector.
     * @param evnt Pointer to a vector of objects of type `g`.
     * @param name Name of the dataset.
     */
    template <typename g>
    void write_event_h5(std::vector<g>* evnt, std::string name){ H5Write(evnt, name); } 

    /**
     * @brief Templated method to write single event data object to HDF5.
     * Specific implementation for event-like data structures.
     * @tparam g Type of the object.
     * @param evnt Pointer to an object of type `g`.
     * @param name Name of the dataset.
     */
    template <typename g>
    void write_event_h5(g* evnt, std::string name){ H5Write(evnt, name); } 

    /**
     * @brief Templated method to read event data (vector of objects) from HDF5.
     * Specific implementation for event-like data structures.
     * @tparam g Type of objects in the vector.
     * @param evnt Pointer to a vector of objects of type `g` to store read data.
     * @param name Name of the dataset.
     */
    template <typename g>
    void read_event_h5(std::vector<g>* evnt, std::string name){ H5Read(evnt, name); } 

    /**
     * @brief Templated method to read single event data object from HDF5.
     * Specific implementation for event-like data structures.
     * @tparam g Type of the object.
     * @param evnt Pointer to an object of type `g` to store read data.
     * @param name Name of the dataset.
     */
    template <typename g>
    void read_event_h5(g* evnt, std::string name){ H5Read(evnt, name); } 

    /**
     * @brief Generic HDF5 write operation for a vector of objects.
     * @tparam g Type of objects in the vector.
     * @param data Pointer to a vector of objects of type `g`.
     * @param set_name Name of the dataset.
     */
    template <typename g>
    void H5Write(std::vector<g>* data, std::string set_name){
        hid_t m = this -> member((*data)[0]); ///< Get HDF5 member type for the first element.
        this -> m_h5 -> template write_hdf5<g>(data, set_name, m); ///< Perform HDF5 write using the obtained member type.
        H5Tclose(m); ///< Close the HDF5 datatype.
    }

    /**
     * @brief Generic HDF5 write operation for a single object.
     * @tparam g Type of the object.
     * @param data Pointer to an object of type `g`.
     * @param set_name Name of the dataset.
     */
    template <typename g>
    void H5Write(g* data, std::string set_name){
        hid_t m = this -> member(*data); ///< Get HDF5 member type for the object.
        this -> m_h5 -> template write_hdf5<g>(data, set_name, m); ///< Perform HDF5 write using the obtained member type.
        H5Tclose(m); ///< Close the HDF5 datatype.
    }

    /**
     * @brief Generic HDF5 read operation for a vector of objects.
     * @tparam g Type of objects in the vector.
     * @param data Pointer to a vector of objects of type `g` to store read data.
     * @param set_name Name of the dataset.
     */
    template <typename g>
    void H5Read(std::vector<g>* data, std::string set_name){
        g el = {}; ///< Create a default-initialized element of type g to get member type.
        hid_t m = this -> member(el); ///< Get HDF5 member type.
        this -> m_h5 -> template read_hdf5<g>(data, set_name, m); ///< Perform HDF5 read.
        H5Tclose(m); ///< Close the HDF5 datatype.
    }

    /**
     * @brief Generic HDF5 read operation for a single object.
     * @tparam g Type of the object.
     * @param data Pointer to an object of type `g` to store read data.
     * @param set_name Name of the dataset.
     */
    template <typename g>
    void H5Read(g* data, std::string set_name){
        hid_t m = this -> member(*data); ///< Get HDF5 member type for the object.
        this -> m_h5 -> template read_hdf5<g>(data, set_name, m); ///< Perform HDF5 read.
        H5Tclose(m); ///< Close the HDF5 datatype.
    }

    std::string current_file = ""; ///< Name of the currently open file. Initialized empty.
    std::string reader_mode = ""; ///< Current reader mode ("read" or "write"). Initialized empty.
    std::string filename_data = ""; ///< Filename associated with data, possibly for metadata. Initialized empty.
    std::string filename_meta = ""; ///< Filename associated with metadata. Initialized empty.

    std::map<std::string, std::string> file_to_access_path = {}; ///< Maps a file identifier to its access path. Initialized empty.
    std::map<std::string, std::vector<std::string>> files_in_dir = {}; ///< Maps directory paths to a list of files within them. Initialized empty.
    std::map<std::string, std::string> files_in_dir_cache = {}; ///< Cache for directory listings. Initialized empty.

    std::map<std::string, data_t*> data = {}; ///< Map storing data read from files, keyed by an identifier. Initialized empty.
    std::map<std::string, TTreeReader*> m_readers = {}; ///< Map storing TTreeReader objects, keyed by an identifier (e.g., tree name). Initialized empty.

    H5::H5File* m_h5_file = nullptr; ///< Pointer to an HDF5 file object. Initialized to nullptr.
    hdf5* m_h5 = nullptr; ///< Pointer to a custom hdf5 handler class instance. Initialized to nullptr.
    settings_t* m_settings = nullptr; ///< Pointer to the settings object. Initialized to nullptr.

    bool m_write = false; ///< Flag indicating if in write mode. Initialized to false.
    bool m_H5event = true; ///< Flag related to HDF5 event processing mode. Initialized to true.
    bool m_use_h5 = false; ///< Flag indicating if HDF5 is being used. Initialized to false.
    bool m_use_root = false; ///< Flag indicating if ROOT is being used. Initialized to false.

}; // End of class 'io' definition.

#endif // End of include guard for IO_IO_H

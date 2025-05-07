/**
 * @file meta.h
 * @brief Handles metadata operations and provides utilities for managing metadata-related tasks.
 * 
 * This file defines the `meta` class, which is responsible for accessing, parsing, and managing 
 * various metadata attributes associated with physics analysis datasets. It includes functionalities 
 * for handling information like Monte Carlo status, event numbers, cross-sections, dataset identifiers, 
 * and more. The class also interfaces with ROOT objects and JSON data for metadata retrieval.
 */

#ifndef META_META_H ///< Start of include guard for META_META_H to prevent multiple inclusions.
#define META_META_H ///< Definition of META_META_H to signify the header has been included.

#include <structs/meta.h> ///< Includes the definition of the `meta_t` structure, which likely holds raw metadata.
#include <structs/folds.h> ///< Includes the definition of the `folds_t` structure, possibly for k-fold cross-validation or data partitioning.
#include <structs/property.h> ///< Includes the definition of the `cproperty` template, a custom property system.
#include <rapidjson/document.h> ///< Includes the RapidJSON library for parsing JSON formatted metadata.
#include <notification/notification.h> ///< Includes the `notification` class, likely for logging or messaging.
#include <tools/tools.h> ///< Includes the `tools` class, providing utility functions.
#include <TFile.h> ///< Includes ROOT's TFile class for file input/output operations.
#include <TTree.h> ///< Includes ROOT's TTree class for handling tree-like data structures.
#include <TBranch.h> ///< Includes ROOT's TBranch class for accessing branches within a TTree.
#include <TLeaf.h> ///< Includes ROOT's TLeaf class for accessing leaves (data elements) within a TBranch.
#include <TH1F.h> ///< Includes ROOT's TH1F class for 1-dimensional histograms with float precision.

/**
 * @class meta
 * @brief Provides properties and methods for metadata management.
 * 
 * The `meta` class inherits from `tools` (providing utility functions) and `notification` 
 * (providing logging/messaging capabilities). It offers a comprehensive interface for 
 * handling various metadata attributes such as event numbers, cross-sections, dataset information,
 * and other parameters crucial for physics analysis. It supports reading metadata from 
 * ROOT files and JSON sources.
 */
class meta : public tools, public notification { // Defines the class 'meta' inheriting from 'tools' and 'notification'.
public: ///< Public access specifier for the following members.
    /**
     * @brief Constructor for the `meta` class.
     * Initializes a new instance of the `meta` class.
     */
    meta();

    /**
     * @brief Destructor for the `meta` class.
     * Cleans up resources used by the `meta` instance, such as the RapidJSON document.
     */
    ~meta();

    /**
     * @brief Retrieves tags based on a hash.
     * @param hash The hash string used to identify and retrieve the specific set of tags.
     * @return A constant pointer to a `folds_t` structure containing the tags, or nullptr if not found.
     */
    const folds_t* get_tags(std::string hash);

    /**
     * @brief Scans data from a generic ROOT object (e.g., TTree, TH1).
     * This method is likely used to extract metadata embedded within ROOT objects.
     * @param obj Pointer to the ROOT TObject to scan for metadata.
     */
    void scan_data(TObject* obj);

    /**
     * @brief Scans sum of weights (SoW) information from a ROOT object.
     * This is often important for normalizing Monte Carlo simulations.
     * @param obj Pointer to the ROOT TObject to scan for sum of weights information.
     */
    void scan_sow(TObject* obj);

    /**
     * @brief Parses a JSON string to extract metadata.
     * @param inpt The JSON string containing metadata to be parsed.
     */
    void parse_json(std::string inpt);

    /**
     * @brief Generates a hash for a given filename.
     * This hash can be used for caching or uniquely identifying file-specific metadata.
     * @param fname The filename (or path) for which to generate the hash.
     * @return The generated hash string.
     */
    std::string hash(std::string fname);

    // Properties: These are cproperty members, a custom property system allowing getter/setter logic.
    cproperty<bool, meta> isMC; ///< Property: Indicates whether the dataset is Monte Carlo (MC) simulation (true) or real data (false).
    cproperty<bool, meta> found; ///< Property: Indicates whether the required metadata was successfully found and loaded.
    cproperty<double, meta> eventNumber; ///< Property: The number of events in the dataset or a specific file.
    cproperty<double, meta> event_index; ///< Property: An index related to events, possibly for iteration or identification.
    cproperty<double, meta> totalSize; ///< Property: The total size of the dataset, potentially in bytes or some other unit.
    cproperty<double, meta> kfactor; ///< Property: The k-factor, a theoretical correction factor used in physics calculations.
    cproperty<double, meta> ecmEnergy; ///< Property: The center-of-mass energy of the collision (e.g., in GeV or TeV).
    cproperty<double, meta> genFiltEff; ///< Property: The generator filter efficiency for MC simulations.
    cproperty<double, meta> completion; ///< Property: A measure of the dataset's processing completion, if applicable.
    cproperty<double, meta> beam_energy; ///< Property: The energy of the colliding beams.

    cproperty<double, meta> cross_section_nb; ///< Property: The physics process cross-section in nanobarns (nb).
    cproperty<double, meta> cross_section_fb; ///< Property: The physics process cross-section in femtobarns (fb).
    cproperty<double, meta> cross_section_pb; ///< Property: The physics process cross-section in picobarns (pb).

    cproperty<double, meta> campaign_luminosity; ///< Property: The integrated luminosity of the data-taking campaign.
    cproperty<double, meta> sum_of_weights; ///< Property: The sum of event weights, crucial for MC normalization.

    cproperty<unsigned int, meta> dsid; ///< Property: The Dataset Identifier (DSID), a unique number for the dataset.
    cproperty<unsigned int, meta> nFiles; ///< Property: The number of files associated with this dataset.
    cproperty<unsigned int, meta> totalEvents; ///< Property: The total number of events across all files in the dataset.
    cproperty<unsigned int, meta> datasetNumber; ///< Property: Another numerical identifier for the dataset, possibly synonymous with DSID.

    cproperty<std::string, meta> derivationFormat; ///< Property: The format of the derived data (e.g., DAOD_PHYS).
    cproperty<std::string, meta> AMITag; ///< Property: The AMI (ATLAS Metadata Interface) tag for the dataset.
    cproperty<std::string, meta> generators; ///< Property: Information about the Monte Carlo event generators used.
    cproperty<std::string, meta> identifier; ///< Property: A generic string identifier for the metadata or dataset.
    cproperty<std::string, meta> DatasetName; ///< Property: The human-readable name of the dataset.
    cproperty<std::string, meta> prodsysStatus; ///< Property: The status of the dataset in the production system.
    cproperty<std::string, meta> dataType; ///< Property: The type of data (e.g., "mc", "data").
    cproperty<std::string, meta> version; ///< Property: Version string for the dataset or metadata schema.
    cproperty<std::string, meta> PDF; ///< Property: Information about the Parton Density Function (PDF) used in MC generation.
    cproperty<std::string, meta> AtlasRelease; ///< Property: The ATLAS software release version used for processing.
    cproperty<std::string, meta> principalPhysicsGroup; ///< Property: The main physics group associated with this dataset (e.g., "SM", "SUSY").
    cproperty<std::string, meta> physicsShort; ///< Property: A short description of the physics process or analysis.
    cproperty<std::string, meta> generatorName; ///< Property: The specific name of the MC event generator.
    cproperty<std::string, meta> geometryVersion; ///< Property: The version of the detector geometry simulation used.
    cproperty<std::string, meta> conditionsTag; ///< Property: The tag for the detector conditions database.
    cproperty<std::string, meta> generatorTune; ///< Property: The specific tune (parameter set) of the MC generator.
    cproperty<std::string, meta> amiStatus; ///< Property: The status of the dataset in the AMI catalog.
    cproperty<std::string, meta> beamType; ///< Property: The type of colliding beams (e.g., "pp", "PbPb").
    cproperty<std::string, meta> productionStep; ///< Property: The step in the data production chain (e.g., "recon", "merge").
    cproperty<std::string, meta> projectName; ///< Property: The name of the project this dataset belongs to (e.g., "mc16_13TeV").
    cproperty<std::string, meta> statsAlgorithm; ///< Property: Algorithm used for statistical combination or analysis.
    cproperty<std::string, meta> genFilterNames; ///< Property: Names of any generator-level filters applied.
    cproperty<std::string, meta> file_type; ///< Property: The type of the file (e.g., "ROOT", "HDF5").
    cproperty<std::string, meta> sample_name; ///< Property: A user-defined name for the sample.
    cproperty<std::string, meta> logicalDatasetName; ///< Property: The Logical Dataset Name (LFN) as used in grid systems.
    cproperty<std::string, meta> campaign; ///< Property: The data-taking or simulation campaign (e.g., "mc16a", "Run2").

    cproperty<std::vector<std::string>, meta> keywords; ///< Property: A list of keywords associated with the dataset.
    cproperty<std::vector<std::string>, meta> weights; ///< Property: A list of weight names or definitions.
    cproperty<std::vector<std::string>, meta> keyword; ///< Property: (Possibly redundant with 'keywords') A list of keywords.
    cproperty<std::vector<std::string>, meta> fileGUID; ///< Property: A list of Globally Unique Identifiers (GUIDs) for the files.

    cproperty<std::vector<int>, meta> events; ///< Property: A list of event counts, possibly per file or run.
    cproperty<std::vector<int>, meta> run_number; ///< Property: A list of run numbers.
    cproperty<std::vector<double>, meta> fileSize; ///< Property: A list of file sizes.

    cproperty<std::map<int, int>, meta> inputrange; ///< Property: A map defining input ranges, possibly for event processing.
    cproperty<std::map<int, std::string>, meta> inputfiles; ///< Property: A map linking an index to input file names/paths.

    cproperty<std::map<std::string, int>, meta> LFN; ///< Property: A map from Logical File Names (LFNs) to an integer (e.g., count or index).
    cproperty<std::map<std::string, weights_t>, meta> misc; ///< Property: A map for miscellaneous weights or related information, using `weights_t` struct.

    cproperty<std::map<std::string, std::string>, meta> config; ///< Property: A map for general configuration settings as key-value string pairs.

private: ///< Private access specifier for the following members.
    rapidjson::Document* rpd = nullptr; ///< Pointer to a RapidJSON document object, used for parsing JSON metadata. Initialized to nullptr.
    std::string metacache_path; ///< String storing the path to a metadata cache, if used.
    meta_t meta_data; ///< Instance of `meta_t` struct, likely holding the raw or processed metadata values.

    friend class analysis; ///< Declares the `analysis` class as a friend, allowing it to access private members of `meta`.

    /**
     * @brief Internal method to compile or process metadata after parsing.
     * This might involve organizing, validating, or transforming the raw metadata.
     */
    void compiler();
    std::vector<folds_t>* folds = nullptr; ///< Pointer to a vector of `folds_t` structures, possibly for k-fold data. Initialized to nullptr.

    /**
     * @brief Parses a float value from a TTree based on a key.
     * @param key The key (branch/leaf name) to look for in the TTree.
     * @param tr Pointer to the TTree to parse from.
     * @return The parsed float value. Returns 0 if not found or on error.
     */
    float parse_float(std::string key, TTree* tr);

    /**
     * @brief Parses a string value from a TTree based on a key.
     * @param key The key (branch/leaf name) to look for in the TTree.
     * @param tr Pointer to the TTree to parse from.
     * @return The parsed string value. Returns empty string if not found or on error.
     */
    std::string parse_string(std::string key, TTree* tr);

    // Static getter methods for cproperty system. These are callbacks used by cproperty.
    /** @brief Static getter for the isMC property.
     *  @param[out] value Pointer to store the retrieved boolean value.
     *  @param[in] instance Pointer to the `meta` instance from which to get the value. */
    static void get_isMC(bool* value, meta* instance);
    /** @brief Static getter for the found property.
     *  @param[out] value Pointer to store the retrieved boolean value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_found(bool* value, meta* instance);
    /** @brief Static getter for the eventNumber property.
     *  @param[out] value Pointer to store the retrieved double value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_eventNumber(double* value, meta* instance);
    /** @brief Static getter for the totalSize property.
     *  @param[out] value Pointer to store the retrieved double value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_totalSize(double* value, meta* instance);
    /** @brief Static getter for the event_index property.
     *  @param[out] value Pointer to store the retrieved double value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_event_index(double* value, meta* instance);
    /** @brief Static getter for the kfactor property.
     *  @param[out] value Pointer to store the retrieved double value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_kfactor(double* value, meta* instance);
    /** @brief Static getter for the ecmEnergy property.
     *  @param[out] value Pointer to store the retrieved double value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_ecmEnergy(double* value, meta* instance);
    /** @brief Static getter for the genFiltEff property.
     *  @param[out] value Pointer to store the retrieved double value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_genFiltEff(double* value, meta* instance);
    /** @brief Static getter for the completion property.
     *  @param[out] value Pointer to store the retrieved double value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_completion(double* value, meta* instance);
    /** @brief Static getter for the beam_energy property.
     *  @param[out] value Pointer to store the retrieved double value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_beam_energy(double* value, meta* instance);

    /** @brief Static getter for the cross_section_pb property.
     *  @param[out] value Pointer to store the retrieved double value (picobarns).
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_cross_section_pb(double* value, meta* instance);
    /** @brief Static getter for the cross_section_nb property.
     *  @param[out] value Pointer to store the retrieved double value (nanobarns).
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_cross_section_nb(double* value, meta* instance);
    /** @brief Static getter for the cross_section_fb property.
     *  @param[out] value Pointer to store the retrieved double value (femtobarns).
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_cross_section_fb(double* value, meta* instance);

    /** @brief Static getter for the campaign_luminosity property.
     *  @param[out] value Pointer to store the retrieved double value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_campaign_luminosity(double* value, meta* instance);
    /** @brief Static getter for the dsid property.
     *  @param[out] value Pointer to store the retrieved unsigned int value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_dsid(unsigned int* value, meta* instance);
    /** @brief Static getter for the nFiles property.
     *  @param[out] value Pointer to store the retrieved unsigned int value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_nFiles(unsigned int* value, meta* instance);
    /** @brief Static getter for the totalEvents property.
     *  @param[out] value Pointer to store the retrieved unsigned int value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_totalEvents(unsigned int* value, meta* instance);
    /** @brief Static getter for the datasetNumber property.
     *  @param[out] value Pointer to store the retrieved unsigned int value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_datasetNumber(unsigned int* value, meta* instance);
    /** @brief Static getter for the derivationFormat property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_derivationFormat(std::string* value, meta* instance);
    /** @brief Static getter for the AMITag property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_AMITag(std::string* value, meta* instance);
    /** @brief Static getter for the generators property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_generators(std::string* value, meta* instance);
    /** @brief Static getter for the identifier property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_identifier(std::string* value, meta* instance);
    /** @brief Static getter for the DatasetName property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_DatasetName(std::string* value, meta* instance);
    /** @brief Static getter for the prodsysStatus property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_prodsysStatus(std::string* value, meta* instance);
    /** @brief Static getter for the dataType property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_dataType(std::string* value, meta* instance);
    /** @brief Static getter for the version property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_version(std::string* value, meta* instance);
    /** @brief Static getter for the PDF property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_PDF(std::string* value, meta* instance);
    /** @brief Static getter for the AtlasRelease property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_AtlasRelease(std::string* value, meta* instance);
    /** @brief Static getter for the principalPhysicsGroup property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_principalPhysicsGroup(std::string* value, meta* instance);
    /** @brief Static getter for the physicsShort property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_physicsShort(std::string* value, meta* instance);
    /** @brief Static getter for the generatorName property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_generatorName(std::string* value, meta* instance);
    /** @brief Static getter for the geometryVersion property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_geometryVersion(std::string* value, meta* instance);
    /** @brief Static getter for the conditionsTag property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_conditionsTag(std::string* value, meta* instance);
    /** @brief Static getter for the generatorTune property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_generatorTune(std::string* value, meta* instance);
    /** @brief Static getter for the amiStatus property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_amiStatus(std::string* value, meta* instance);
    /** @brief Static getter for the beamType property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_beamType(std::string* value, meta* instance);
    /** @brief Static getter for the productionStep property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_productionStep(std::string* value, meta* instance);
    /** @brief Static getter for the projectName property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_projectName(std::string* value, meta* instance);
    /** @brief Static getter for the statsAlgorithm property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_statsAlgorithm(std::string* value, meta* instance);
    /** @brief Static getter for the genFilterNames property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_genFilterNames(std::string* value, meta* instance);
    /** @brief Static getter for the file_type property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_file_type(std::string* value, meta* instance);
    /** @brief Static getter for the sample_name property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_sample_name(std::string* value, meta* instance);
    /** @brief Static getter for the logicalDatasetName property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_logicalDatasetName(std::string* value, meta* instance);
    /** @brief Static getter for the campaign property.
     *  @param[out] value Pointer to store the retrieved string value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_campaign(std::string* value, meta* instance);
    /** @brief Static getter for the keywords property.
     *  @param[out] value Pointer to store the retrieved vector of strings.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_keywords(std::vector<std::string>* value, meta* instance);
    /** @brief Static getter for the weights property.
     *  @param[out] value Pointer to store the retrieved vector of strings.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_weights(std::vector<std::string>* value, meta* instance);
    /** @brief Static getter for the keyword property.
     *  @param[out] value Pointer to store the retrieved vector of strings.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_keyword(std::vector<std::string>* value, meta* instance);
    /** @brief Static getter for the fileGUID property.
     *  @param[out] value Pointer to store the retrieved vector of strings (GUIDs).
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_fileGUID(std::vector<std::string>* value, meta* instance);
    /** @brief Static getter for the events property.
     *  @param[out] value Pointer to store the retrieved vector of integers.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_events(std::vector<int>* value, meta* instance);
    /** @brief Static getter for the run_number property.
     *  @param[out] value Pointer to store the retrieved vector of integers.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_run_number(std::vector<int>* value, meta* instance);
    /** @brief Static getter for the fileSize property.
     *  @param[out] value Pointer to store the retrieved vector of doubles.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_fileSize(std::vector<double>* value, meta* instance);

    /** @brief Static getter for the inputrange property.
     *  @param[out] value Pointer to store the retrieved map of int to int.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_inputrange(std::map<int, int>* value, meta* instance);
    /** @brief Static getter for the inputfiles property.
     *  @param[out] value Pointer to store the retrieved map of int to string.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_inputfiles(std::map<int, std::string>* value, meta* instance);

    /** @brief Static getter for the LFN property.
     *  @param[out] value Pointer to store the retrieved map of string to int.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_LFN(std::map<std::string, int>* value, meta* instance);
    /** @brief Static getter for the misc property.
     *  @param[out] value Pointer to store the retrieved map of string to `weights_t`.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_misc(std::map<std::string, weights_t>* value, meta* instance);

    /** @brief Static getter for the config property.
     *  @param[out] value Pointer to store the retrieved map of string to string.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_config(std::map<std::string, std::string>* value, meta* instance);
    /** @brief Static getter for the sum_of_weights property.
     *  @param[out] value Pointer to store the retrieved double value.
     *  @param[in] instance Pointer to the `meta` instance. */
    static void get_sum_of_weights(double* value, meta* instance);
}; // End of class 'meta' definition.

#endif // META_META_H ///< End of include guard for META_META_H.

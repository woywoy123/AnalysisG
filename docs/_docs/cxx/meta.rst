.. cpp:class:: meta
    :inherits: tools, notification

    @brief Manages and provides access to metadata for physics analysis datasets.

    This class serves as a central hub for handling metadata associated with high-energy physics datasets. It is designed to extract metadata from various sources, primarily ROOT files (TTrees like "MetaData", "AnalysisTracking", and sum-of-weights histograms) and JSON configuration strings. It consolidates this information into an internal `meta_t` structure and provides a consistent and convenient interface for accessing these metadata fields.

    Key functionalities include:
    - Parsing JSON metadata strings using `rapidjson`.
    - Scanning ROOT `TObject`s (TTrees, TH1s) to extract relevant information like event counts, sum-of-weights, dataset identifiers, campaign details, and generator information.
    - Storing extracted metadata in the `meta_data` member.
    - Providing access to metadata fields through `cproperty` objects, which enable lazy evaluation and a clean syntax (e.g., `meta_object.dsid`). These properties retrieve values on demand using dedicated static getter functions.
    - Calculating derived metadata, such as cross-sections in different units (pb, fb) from the base value (nb).
    - Handling potential metadata caching (though the mechanism isn't fully detailed in the provided code).
    - Managing cross-validation fold information via the `folds` member.

    The class inherits from `tools` (likely providing utility functions like string manipulation and hashing) and `notification` (likely providing logging or messaging capabilities).

    @publicsection Public Members
    .. cpp:member:: rapidjson::Document* rpd = nullptr

        @brief Pointer to the `rapidjson::Document` used for parsing JSON metadata.
        This pointer is managed internally by the `parse_json` method. It is allocated when `parse_json` is called and deleted after the JSON data has been processed by the `compiler` method or if parsing fails. It remains `nullptr` otherwise.

    .. cpp:member:: std::string metacache_path

        @brief Path specified for a potential metadata cache file.
        The exact usage and implementation details of the caching mechanism based on this path are not fully evident in the provided code snippets but suggest an intended feature for storing/retrieving metadata to/from a file.

    .. cpp:member:: meta_t meta_data

        @brief Structure holding the raw extracted or parsed metadata values.
        This member contains the actual data fields populated by methods like `scan_data`, `scan_sow`, and `compiler`. The `cproperty` members access the fields within this structure via their getter functions.
        @see :cpp:struct:`meta_t` for the definition of the fields.

    .. cpp:member:: cproperty<bool, meta> isMC

        @brief Property: Indicates if the dataset is Monte Carlo (MC) simulation (`true`) or real data (`false`).
        Accessed via `get_isMC`.

    .. cpp:member:: cproperty<bool, meta> found

        @brief Property: Indicates if the metadata was successfully located and loaded.
        Accessed via `get_found`. (Note: The specific logic setting this flag isn't fully shown).

    .. cpp:member:: cproperty<double, meta> eventNumber

        @brief Property: Represents the event number, likely corresponding to the current event being processed in an event loop.
        Accessed via `get_eventNumber`.

    .. cpp:member:: cproperty<double, meta> event_index

        @brief Property: Represents the index of the current event within its file or the dataset context.
        Accessed via `get_event_index`.

    .. cpp:member:: cproperty<double, meta> totalSize

        @brief Property: Represents the total size of the dataset or associated files, potentially in bytes.
        Accessed via `get_totalSize`.

    .. cpp:member:: cproperty<double, meta> kfactor

        @brief Property: Represents the k-factor applied for theoretical corrections (e.g., NLO/LO scaling).
        Accessed via `get_kfactor`.

    .. cpp:member:: cproperty<double, meta> ecmEnergy

        @brief Property: Represents the center-of-mass energy of the collision (e.g., 13000 for 13 TeV).
        Accessed via `get_ecmEnergy`.

    .. cpp:member:: cproperty<double, meta> genFiltEff

        @brief Property: Represents the generator-level filter efficiency applied during MC event generation.
        Accessed via `get_genFiltEff`.

    .. cpp:member:: cproperty<double, meta> completion

        @brief Property: Represents the completion status or fraction of the dataset processing.
        Accessed via `get_completion`.

    .. cpp:member:: cproperty<double, meta> beam_energy

        @brief Property: Represents the energy of the individual colliding beams.
        Accessed via `get_beam_energy`.

    .. cpp:member:: cproperty<double, meta> cross_section_nb

        @brief Property: Represents the physics process cross-section in nanobarns (nb). This is often the base unit stored in metadata.
        Accessed via `get_cross_section_nb`.

    .. cpp:member:: cproperty<double, meta> cross_section_fb

        @brief Property: Represents the physics process cross-section in femtobarns (fb). Derived from `cross_section_nb`.
        Accessed via `get_cross_section_fb`. (Calculated as nb * 1e6).

    .. cpp:member:: cproperty<double, meta> cross_section_pb

        @brief Property: Represents the physics process cross-section in picobarns (pb). Derived from `cross_section_nb`.
        Accessed via `get_cross_section_pb`. (Calculated as nb * 1e3).

    .. cpp:member:: cproperty<double, meta> campaign_luminosity

        @brief Property: Represents the integrated luminosity corresponding to the data-taking period or campaign.
        Accessed via `get_campaign_luminosity`.

    .. cpp:member:: cproperty<double, meta> sum_of_weights

        @brief Property: Represents the total sum of event weights for the dataset. Crucial for normalizing Monte Carlo samples.
        This value is typically extracted from sum-of-weights histograms or dedicated metadata trees.
        Accessed via `get_sum_of_weights`.

    .. cpp:member:: cproperty<unsigned int, meta> dsid

        @brief Property: Represents the Dataset Identifier (DSID), a unique number identifying the dataset.
        Accessed via `get_dsid`.

    .. cpp:member:: cproperty<unsigned int, meta> nFiles

        @brief Property: Represents the number of files constituting the dataset.
        Accessed via `get_nFiles`.

    .. cpp:member:: cproperty<unsigned int, meta> totalEvents

        @brief Property: Represents the total number of events in the dataset *before* any skimming or filtering applied in the current processing.
        Accessed via `get_totalEvents`.

    .. cpp:member:: cproperty<unsigned int, meta> datasetNumber

        @brief Property: Represents the dataset number, often synonymous with the DSID.
        Accessed via `get_datasetNumber`.

    .. cpp:member:: cproperty<std::string, meta> derivationFormat

        @brief Property: Represents the format of the derived data (e.g., "DAOD_PHYS", "DAOD_PHYSLITE").
        Accessed via `get_derivationFormat`.

    .. cpp:member:: cproperty<std::string, meta> AMITag

        @brief Property: Represents the AMI (ATLAS Metadata Interface) tag associated with the dataset production configuration.
        Accessed via `get_AMITag`.

    .. cpp:member:: cproperty<std::string, meta> generators

        @brief Property: Represents the event generator(s) used for producing the MC simulation (e.g., "Pythia8", "Sherpa").
        Accessed via `get_generators`.

    .. cpp:member:: cproperty<std::string, meta> identifier

        @brief Property: Represents a unique identifier string for the dataset or the specific processing job.
        Accessed via `get_identifier`.

    .. cpp:member:: cproperty<std::string, meta> DatasetName

        @brief Property: Represents the full, official name of the dataset.
        Accessed via `get_DatasetName`.

    .. cpp:member:: cproperty<std::string, meta> prodsysStatus

        @brief Property: Represents the status within the production system (e.g., "completed", "running").
        Accessed via `get_prodsysStatus`.

    .. cpp:member:: cproperty<std::string, meta> dataType

        @brief Property: Represents the type of data, typically "mc" for Monte Carlo or "data" for real collision data.
        Accessed via `get_dataType`.

    .. cpp:member:: cproperty<std::string, meta> version

        @brief Property: Represents the version identifier for the dataset production or processing.
        Accessed via `get_version`.

    .. cpp:member:: cproperty<std::string, meta> PDF

        @brief Property: Represents the Parton Density Function (PDF) set used in the MC generation.
        Accessed via `get_PDF`.

    .. cpp:member:: cproperty<std::string, meta> AtlasRelease

        @brief Property: Represents the version of the ATLAS software release used for production or analysis.
        Accessed via `get_AtlasRelease`.

    .. cpp:member:: cproperty<std::string, meta> principalPhysicsGroup

        @brief Property: Represents the main ATLAS physics working group associated with this dataset (e.g., "Top", "Higgs", "Exotics").
        Accessed via `get_principalPhysicsGroup`.

    .. cpp:member:: cproperty<std::string, meta> physicsShort

        @brief Property: Represents a short name or code identifying the physics process simulated or analyzed.
        Accessed via `get_physicsShort`.

    .. cpp:member:: cproperty<std::string, meta> generatorName

        @brief Property: Represents the name of the primary event generator software.
        Accessed via `get_generatorName`.

    .. cpp:member:: cproperty<std::string, meta> geometryVersion

        @brief Property: Represents the version identifier for the detector geometry simulation used.
        Accessed via `get_geometryVersion`.

    .. cpp:member:: cproperty<std::string, meta> conditionsTag

        @brief Property: Represents the tag identifying the set of detector conditions (calibration, alignment) used.
        Accessed via `get_conditionsTag`.

    .. cpp:member:: cproperty<std::string, meta> generatorTune

        @brief Property: Represents the specific set of parameters (tune) used to configure the event generator.
        Accessed via `get_generatorTune`.

    .. cpp:member:: cproperty<std::string, meta> amiStatus

        @brief Property: Represents the status of the dataset as recorded in the AMI database.
        Accessed via `get_amiStatus`.

    .. cpp:member:: cproperty<std::string, meta> beamType

        @brief Property: Represents the type of particles collided (e.g., "pp" for proton-proton, "HI" for heavy ion).
        Accessed via `get_beamType`.

    .. cpp:member:: cproperty<std::string, meta> productionStep

        @brief Property: Represents the stage in the data processing chain (e.g., "simul", "recon", "deriv", "merge").
        Accessed via `get_productionStep`.

    .. cpp:member:: cproperty<std::string, meta> projectName

        @brief Property: Represents the name of the overall production project (e.g., "mc16_13TeV").
        Accessed via `get_projectName`.

    .. cpp:member:: cproperty<std::string, meta> statsAlgorithm

        @brief Property: Represents the algorithm used for statistical combinations or analyses.
        Accessed via `get_statsAlgorithm`.

    .. cpp:member:: cproperty<std::string, meta> genFilterNames

        @brief Property: Represents the names of any generator-level filters applied during event generation.
        Accessed via `get_genFilterNames`.

    .. cpp:member:: cproperty<std::string, meta> file_type

        @brief Property: Represents the type or format of the input file (e.g., "ROOT", "NTUP", "DAOD").
        Accessed via `get_file_type`.

    .. cpp:member:: cproperty<std::string, meta> sample_name

        @brief Property: Represents the logical name assigned to the sample or dataset.
        Accessed via `get_sample_name`.

    .. cpp:member:: cproperty<std::string, meta> logicalDatasetName

        @brief Property: Represents the Logical Dataset Name (LDN or LFN for Logical File Name context).
        Accessed via `get_logicalDatasetName`.

    .. cpp:member:: cproperty<std::string, meta> campaign

        @brief Property: Represents the specific data-taking period (e.g., "data18") or MC simulation campaign (e.g., "mc16a", "mc16d", "mc16e").
        Accessed via `get_campaign`.

    .. cpp:member:: cproperty<std::vector<std::string>, meta> keywords

        @brief Property: Represents a list of keywords associated with the dataset, often used for categorization or searching.
        Accessed via `get_keywords`.

    .. cpp:member:: cproperty<std::vector<std::string>, meta> weights

        @brief Property: Represents a list of names corresponding to systematic uncertainty weights available in the dataset.
        Accessed via `get_weights`.

    .. cpp:member:: cproperty<std::vector<std::string>, meta> keyword

        @brief Property: Represents a list of keywords (potentially redundant with `keywords` or used for a different purpose).
        Accessed via `get_keyword`.

    .. cpp:member:: cproperty<std::vector<std::string>, meta> fileGUID

        @brief Property: Represents a list of Globally Unique Identifiers (GUIDs) for the individual files within the dataset.
        Accessed via `get_fileGUID`.

    .. cpp:member:: cproperty<std::vector<int>, meta> events

        @brief Property: Represents a list of event counts, possibly on a per-file basis.
        Accessed via `get_events`.

    .. cpp:member:: cproperty<std::vector<int>, meta> run_number

        @brief Property: Represents a list of run numbers included in the dataset.
        Accessed via `get_run_number`.

    .. cpp:member:: cproperty<std::vector<double>, meta> fileSize

        @brief Property: Represents a list of file sizes, likely corresponding to the individual files in the dataset.
        Accessed via `get_fileSize`.

    .. cpp:member:: cproperty<std::map<int, int>, meta> inputrange

        @brief Property: Represents a map defining input ranges. The exact meaning (e.g., event ranges per file, run ranges) is context-dependent.
        Accessed via `get_inputrange`.

    .. cpp:member:: cproperty<std::map<int, std::string>, meta> inputfiles

        @brief Property: Represents a map linking an index (potentially cumulative event count) to input file names (often basenames). Populated from JSON or specific TTrees.
        Accessed via `get_inputfiles`.

    .. cpp:member:: cproperty<std::map<std::string, int>, meta> LFN

        @brief Property: Represents a map linking Logical File Names (LFNs) to an integer value (e.g., event count in that file).
        Accessed via `get_LFN`.

    .. cpp:member:: cproperty<std::map<std::string, weights_t>, meta> misc

        @brief Property: Represents a map storing miscellaneous metadata, particularly sum-of-weights information extracted from different sources (histograms, trees). The key is often the name of the source object (e.g., histogram name), and the value is a `weights_t` struct containing detailed weight information.
        Accessed via `get_misc`.
        @see :cpp:struct:`weights_t`

    .. cpp:member:: cproperty<std::map<std::string, std::string>, meta> config

        @brief Property: Represents a map storing configuration key-value pairs used during the data processing or analysis job setup.
        Accessed via `get_config`.

    @publicsection Public Functions
    .. cpp:function:: meta()

        @brief Default constructor.
        Initializes the `meta` object. It sets up all the `cproperty` members by assigning their respective static getter functions (`get_...`) and associating them with the current object instance (`this`) using `set_getter` and `set_object`. It also sets the default prefix for notification messages inherited from the `notification` base class to "meta".

    .. cpp:function:: ~meta()

        @brief Destructor.
        Handles resource cleanup. Specifically, it checks if the `rapidjson::Document` pointer `rpd` is non-null (meaning it was allocated during `parse_json`) and deletes the allocated document to prevent memory leaks.

    .. cpp:function:: const folds_t* get_tags(std::string hash_)

        @brief Retrieves cross-validation fold information associated with a given hash.
        Searches the internal `folds` collection (if it's not null) for an entry whose `hash` member matches the provided `hash_` string. This is likely used to determine which cross-validation fold a particular event or file belongs to based on a precomputed hash.

        @param hash_ The hash string (likely generated from event or file identifiers) to search for.
        @return A constant pointer to the `folds_t` struct containing the fold information if a match is found; otherwise, returns `nullptr`.
        @see folds_t
        @see hash()

    .. cpp:function:: void scan_data(TObject* obj)

        @brief Scans a generic ROOT TObject to extract metadata, dispatching to specific handlers.
        This function acts as an entry point for extracting metadata from ROOT objects obtained from input files. It inspects the type and name of the `obj`:
        - If `obj` is a TTree named "AnalysisTracking", it assumes the metadata is stored as a JSON string in the "jsonData" branch, extracts it using `parse_string`, and passes it to `parse_json`.
        - If `obj` is a TTree named "MetaData", it assumes the tree structure directly maps to the `meta_data` member and sets the branch address accordingly to populate `meta_data` when the tree is read.
        - For any other object type or name, it calls `scan_sow` to attempt extraction of sum-of-weights information from histograms or other recognized tree structures.
        It temporarily sets `gErrorIgnoreLevel` high to suppress informational messages from ROOT during object scanning.

        @param obj Pointer to the TObject (e.g., TTree, TH1) from the input file to be scanned for metadata.

    .. cpp:function:: void scan_sow(TObject* obj)

        @brief Scans a TObject, primarily targeting histograms and specific TTrees, for sum-of-weights (SOW) and related metadata.
        This function specializes in extracting metadata, focusing on sum-of-weights values, event counts, and basic dataset identifiers, often found in histograms or specific TTrees added during processing steps.
        - If `obj` is a TTree:
          - If named "AnalysisTracking": Extracts various fields like DSID, event counts (total, processed, weighted), generator name, and AMI tag using `parse_float` and `parse_string` for the first entry. Stores results in a `weights_t` struct within `meta_data.misc` keyed by the tree name.
          - If named "EventLoop_FileExecuted": Reads the list of executed file names (TString) and stores them in `meta_data.inputfiles`, mapping index to filename.
        - If `obj` inherits from TH1 (specifically handled as TH1F):
          - Iterates through the histogram bins, extracting content based on specific bin labels (e.g., "Initial events", "Initial sum of weights", "Initial sum of weights squared").
          - Attempts to identify campaign, DSID, and AMI tag from specific bin labels if a campaign label (containing "mc") is found.
          - Stores the extracted histogram bin labels and contents in the `hist_data` map within a `weights_t` struct in `meta_data.misc`, keyed by the histogram name.

        @param obj Pointer to the TObject (typically TH1F or TTree) to scan.

    .. cpp:function:: void parse_json(std::string inpt)

        @brief Parses a JSON formatted string to populate metadata fields.
        Takes a string `inpt` assumed to contain metadata in JSON format.
        - Allocates a `rapidjson::Document` pointed to by `rpd`.
        - Attempts to parse the `inpt` string using `rpd->Parse()`.
        - If parsing fails, it attempts a simple heuristic fix: if the error occurs near a newline that isn't preceded by a comma, it replaces the newline with a comma-newline sequence in the input string and retries parsing with a new `rapidjson::Document`. This handles a common JSON formatting issue.
        - If parsing is successful (either initially or after the fix), it calls the private `compiler` method to extract data from the parsed `rpd` document into `meta_data`.
        - Finally, it deletes the allocated `rapidjson::Document` pointed to by `rpd` and resets `rpd` to `nullptr`.
        @note This function manages the lifecycle of the `rpd` pointer during its execution. It returns early if `rpd` is already allocated, suggesting it's not designed for re-entrant parsing without destruction.

        @param inpt The std::string containing the metadata in JSON format.

    .. cpp:function:: std::string hash(std::string fname)

        @brief Generates a hash string, typically derived from the basename of a file path.
        Takes an input string `fname`, which is usually expected to be a file path. It splits the string by the '/' character. If '/' is present, it calculates a hash (using the inherited `tools::hash` method) of the last component (the filename). If `fname` does not contain '/', it hashes the entire input string. This is likely used to generate consistent identifiers for files, potentially for use with `get_tags`.

        @param fname The input string, typically a full file path or filename.
        @return A hash string computed from the filename component of `fname` or the full `fname` if no '/' is present.

    @privatesection Private Members
    .. cpp:member:: std::vector<folds_t>* folds = nullptr

        @brief Pointer to a vector containing cross-validation fold definitions.
        This pointer likely holds information mapping hashes (potentially generated by the `hash` method) to cross-validation fold indices (training/validation/testing). It is expected to be populated externally, possibly by an analysis steering class (`friend analysis`).
        @see folds_t
        @see get_tags()

    @privatesection Private Functions
    .. cpp:function:: void compiler()

        @brief Processes the parsed JSON data (`rpd`) to populate `meta_data`.
        This internal helper function is called by `parse_json` after a JSON string has been successfully parsed into the `rpd` document. It navigates the JSON structure:
        - Extracts values from the "inputConfig" object (e.g., "dsid", "isMC", "derivationFormat", "amiTag") and assigns them to the corresponding fields in `meta_data`.
        - Processes the "configSettings" array (if present) and populates the `meta_data.config` map.
        - Iterates through the "inputFiles" array, extracting filenames and event counts, populating `meta_data.inputfiles` (mapping cumulative event count index to filename basename).
        - Includes logic to determine the `AMITag` from "inputConfig", or by parsing it from the directory structure of the input file paths listed in "inputFiles" if not directly available in "inputConfig". It might also fall back to parsing the `sample_name` if other methods fail.

    .. cpp:function:: float parse_float(std::string key, TTree* tr)

        @brief Utility function to extract a single float value from a TTree leaf in the first entry.
        Reads the value of the leaf named `key` from the first entry (entry 0) of the provided TTree `tr`. Assumes the leaf contains a float-compatible type.

        @param key The name of the TLeaf (branch) containing the float value.
        @param tr Pointer to the TTree to read from.
        @return The float value read from the specified leaf in the first entry. Returns 0 if the leaf is not found or cannot be read.

    .. cpp:function:: std::string parse_string(std::string key, TTree* tr)

        @brief Utility function to extract a string value from a TTree branch (potentially spanning multiple leaves) in the first entry.
        Reads the string data associated with the branch named `key` from the first entry (entry 0) of the TTree `tr`. It handles cases where a long string might be stored across multiple TLeaf objects within the same TBranch by concatenating their contents.

        @param key The name of the TBranch containing the string value.
        @param tr Pointer to the TTree to read from.
        @return The string value read from the specified branch in the first entry. Returns an empty string if the branch is not found or cannot be read.

    .. cpp:function:: void static get_sum_of_weights(double* val, meta* m)

        @brief Static getter for the `sum_of_weights` property.
        Retrieves the sum of weights. It iterates through the `weights_t` entries stored in the `meta_data.misc` map (which are populated by `scan_sow`). It returns the first non-negative `processed_events_weighted` value found. This accommodates scenarios where sum-of-weights might be stored in different histograms or trees.

        @param[out] val Pointer to the double where the retrieved sum of weights will be stored.
        @param m Pointer to the `meta` object instance whose `meta_data` should be accessed.

    .. cpp:function:: void static get_cross_section_fb(double* val, meta* m)

        @brief Static getter for the `cross_section_fb` property.
        Calculates the cross-section in femtobarns (fb) by retrieving the nanobarn value (via the `cross_section_nb` property) and multiplying it by 1,000,000.

        @param[out] val Pointer to the double where the calculated cross-section in fb will be stored.
        @param m Pointer to the `meta` object instance.

    .. cpp:function:: void static get_cross_section_pb(double* val, meta* m)

        @brief Static getter for the `cross_section_pb` property.
        Calculates the cross-section in picobarns (pb) by retrieving the nanobarn value (via the `cross_section_nb` property) and multiplying it by 1,000.

        @param[out] val Pointer to the double where the calculated cross-section in pb will be stored.
        @param m Pointer to the `meta` object instance.

    .. cpp:function:: void static get_cross_section_nb(double* val, meta* m)

        @brief Static getter for the `cross_section_nb` property.
        Retrieves the cross-section value directly from `m->meta_data.crossSection_mean`. It assumes this field stores the value in nanobarns (nb).

        @param[out] val Pointer to the double where the retrieved cross-section in nb will be stored.
        @param m Pointer to the `meta` object instance whose `meta_data` should be accessed.

    .. cpp:function:: void static get_campaign(std::string* val, meta* m)

        @brief Static getter for the `campaign` property.
        Retrieves the campaign string from `m->meta_data.campaign`. It first removes any spaces from the stored campaign string before assigning it to `*val`.

        @param[out] val Pointer to the std::string where the retrieved campaign name will be stored.
        @param m Pointer to the `meta` object instance whose `meta_data` should be accessed.

    .. note::
        The following static getter functions provide direct access to the corresponding fields within the `meta_data` member for their respective `cproperty` counterparts. They all follow the same pattern: retrieve the value from `m->meta_data.fieldName` and store it in `*val`.

        - ``void static get_isMC(bool* val, meta* m)``
        - ``void static get_found(bool* val, meta* m)``
        - ``void static get_eventNumber(double* val, meta* m)``
        - ``void static get_event_index(double* val, meta* m)``
        - ``void static get_totalSize(double* val, meta* m)``
        - ``void static get_kfactor(double* val, meta* m)``
        - ``void static get_ecmEnergy(double* val, meta* m)``
        - ``void static get_genFiltEff(double* val, meta* m)``
        - ``void static get_completion(double* val, meta* m)``
        - ``void static get_beam_energy(double* val, meta* m)``
        - ``void static get_campaign_luminosity(double* val, meta* m)``
        - ``void static get_dsid(unsigned int* val, meta* m)``
        - ``void static get_nFiles(unsigned int* val, meta* m)``
        - ``void static get_totalEvents(unsigned int* val, meta* m)``
        - ``void static get_datasetNumber(unsigned int* val, meta* m)``
        - ``void static get_derivationFormat(std::string* val, meta* m)``
        - ``void static get_AMITag(std::string* val, meta* m)``
        - ``void static get_generators(std::string* val, meta* m)``
        - ``void static get_identifier(std::string* val, meta* m)``
        - ``void static get_DatasetName(std::string* val, meta* m)``
        - ``void static get_prodsysStatus(std::string* val, meta* m)``
        - ``void static get_dataType(std::string* val, meta* m)``
        - ``void static get_version(std::string* val, meta* m)``
        - ``void static get_PDF(std::string* val, meta* m)``
        - ``void static get_AtlasRelease(std::string* val, meta* m)``
        - ``void static get_principalPhysicsGroup(std::string* val, meta* m)``
        - ``void static get_physicsShort(std::string* val, meta* m)``
        - ``void static get_generatorName(std::string* val, meta* m)``
        - ``void static get_geometryVersion(std::string* val, meta* m)``
        - ``void static get_conditionsTag(std::string* val, meta* m)``
        - ``void static get_generatorTune(std::string* val, meta* m)``
        - ``void static get_amiStatus(std::string* val, meta* m)``
        - ``void static get_beamType(std::string* val, meta* m)``
        - ``void static get_productionStep(std::string* val, meta* m)``
        - ``void static get_projectName(std::string* val, meta* m)``
        - ``void static get_statsAlgorithm(std::string* val, meta* m)``
        - ``void static get_genFilterNames(std::string* val, meta* m)``
        - ``void static get_file_type(std::string* val, meta* m)``
        - ``void static get_sample_name(std::string* val, meta* m)``
        - ``void static get_logicalDatasetName(std::string* val, meta* m)``
        - ``void static get_keywords(std::vector<std::string>* val, meta* m)``
        - ``void static get_weights(std::vector<std::string>* val, meta* m)``
        - ``void static get_keyword(std::vector<std::string>* val, meta* m)``
        - ``void static get_fileGUID(std::vector<std::string>* val, meta* m)``
        - ``void static get_events(std::vector<int>* val, meta* m)``
        - ``void static get_run_number(std::vector<int>* val, meta* m)``
        - ``void static get_fileSize(std::vector<double>* val, meta* m)``
        - ``void static get_inputrange(std::map<int, int>* val, meta* m)``
        - ``void static get_inputfiles(std::map<int, std::string>* val, meta* m)``
        - ``void static get_LFN(std::map<std::string, int>* val, meta* m)``
        - ``void static get_misc(std::map<std::string, weights_t>* val, meta* m)``
        - ``void static get_config(std::map<std::string, std::string>* val, meta* m)``

};

#endif

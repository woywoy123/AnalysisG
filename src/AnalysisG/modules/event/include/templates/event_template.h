/**
 * @file event_template.h
 * @brief Defines the base template class for event data representation and manipulation.
 *
 * This file contains the declaration of the `event_template` class, which serves as the base template
 * for processing physics event data. It provides functionality for managing trees, branches, and leaves
 * in physics data structures, as well as properties like event weight and index. The class also supports
 * dynamic particle registration and building events from raw data.
 *
 * The event architecture is designed to be flexible and extensible, allowing customization
 * for different types of physics analyses and data formats.
 *
 * @section event_template_usage Usage
 *
 * Subclass event_template to create custom event types:
 *
 * ```cpp
 * class MyEvent : public event_template {
 * public:
 *     MyEvent() : event_template() {
 *         add_tree("nominal");
 *         add_branch("el_");    // Electron branches
 *         add_branch("mu_");    // Muon branches  
 *         add_branch("jet_");   // Jet branches
 *         add_leaf("met_et");
 *         add_leaf("met_phi");
 *         
 *         // Register particle types
 *         register_particle("Electron", "el_");
 *         register_particle("Muon", "mu_");
 *         register_particle("Jet", "jet_");
 *     }
 *     
 *     void build(element_t* el) override {
 *         // Custom event building logic
 *     }
 * };
 * ```
 */

#ifndef EVENT_TEMPLATE_H ///< Start of include guard for EVENT_TEMPLATE_H to prevent multiple inclusions.
#define EVENT_TEMPLATE_H ///< Definition of EVENT_TEMPLATE_H to indicate the header has been included.

#include <templates/particle_template.h> ///< Includes the `particle_template` class for particle data handling.
#include <structs/property.h> ///< Includes the `cproperty` template for custom property management.
#include <structs/element.h> ///< Includes the `element_t` structure, a base data element type.
#include <structs/event.h> ///< Includes the `event_t` structure for event data representation.
#include <tools/tools.h> ///< Includes the `tools` class for utility functions.
#include <meta/meta.h> ///< Includes the `meta` class for metadata handling.

/**
 * @class event_template
 * @brief Base template class for event data representation and manipulation.
 *
 * Inherits from `tools` to provide utility functions. This class is designed to be subclassed
 * for specific event types or analyses. It manages collections of trees, branches, and leaves
 * that define the event structure, as well as properties such as event weight and index.
 * It supports dynamic particle registration and building events from raw data.
 *
 * The class serves as a central component in the AnalysisG framework, connecting raw data
 * with higher-level analysis objects and algorithms. It provides interfaces for data access,
 * filtering, and transformation needed for physics analyses.
 *
 * @section event_template_structure Data Structure
 *
 * Events are organized hierarchically:
 * - **Trees**: ROOT TTrees containing the data (e.g., "nominal", "systematic")
 * - **Branches**: Particle collections within trees (e.g., "el_", "jet_")
 * - **Leaves**: Individual variables (e.g., "el_pt", "met_et")
 *
 * @section event_template_particles Particle Management
 *
 * Particles are registered using `register_particle()` and accessed via `particle_link`:
 * - Each particle type maps to a collection of particle_template instances
 * - Particles are built from raw data during `build()` calls
 */
class event_template: public tools
{
public:
    /**
     * @brief Default constructor for event_template.
     *
     * Initializes an empty event template with default properties and settings.
     * Sets up property mappings and initializes collections.
     */
    event_template();

    /**
     * @brief Destructor for event_template.
     *
     * Cleans up allocated resources including registered particles and data structures.
     */
    virtual ~event_template();

    /**
     * @brief Creates a new event template instance.
     * @return Pointer to the newly created event_template object.
     *
     * Factory method that creates and returns a new instance of the event_template class.
     * Useful for creating derived class instances in a polymorphic context.
     */
    virtual event_template* clone();

    /**
     * @brief Resets the event template to its initial state.
     *
     * Clears all event data and prepares the template for new event processing.
     */
    virtual void reset();

    /**
     * @brief Initializes the event template with necessary settings.
     *
     * Sets up internal data structures and configurations. This method should be called
     * before performing other operations on the template.
     */
    void initialize();

    /**
     * @brief Builds an event from raw data.
     *
     * Virtual method that processes raw data to create a structured event representation.
     * This base method delegates implementation to the overloaded build(element_t*) method.
     */
    void build();

    /**
     * @brief Builds an event data structure from an element.
     * @param el Pointer to an element_t structure containing raw event data.
     *
     * Virtual method that processes raw element data to build the event structure.
     * This method is typically overridden in derived classes to handle specific event formats.
     */
    virtual void build(element_t* el);

    /**
     * @brief Builds a mapping between event data structures and handlers.
     * @param evnt Pointer to a map associating strings with data_t pointers.
     *
     * Creates relationships between the event's trees, branches, leaves and their
     * corresponding data handlers to facilitate data access and manipulation.
     */
    void build_mapping(std::map<std::string, data_t*>* evnt);

    /**
     * @brief Clears all leaf string references.
     *
     * Resets internal collections of trees, branches, and leaves.
     * This is useful when preparing to process a new event format.
     */
    void clear_leaves();

    /**
     * @brief Registers a particle for tracking in this event.
     * @param name The name that identifies this particle type (e.g., "Electron").
     * @param collection The branch prefix for this particle collection (e.g., "el_").
     */
    void register_particle(std::string name, std::string collection);

    /**
     * @brief Adds a tree to the event structure.
     * @param key The primary key or name for this tree.
     * @param tree The specific tree name if different from key, or an alias.
     */
    void add_tree(std::string key, std::string tree = "");

    /**
     * @brief Adds a branch to the event structure.
     * @param key The primary key or name for this branch (e.g., "el_").
     * @param branch The specific branch name if different from key, or an alias.
     */
    void add_branch(std::string key, std::string branch = "");

    /**
     * @brief Adds a leaf (variable) to the event structure.
     * @param key The primary key or name for this leaf (e.g., "met_et").
     * @param leaf The specific leaf name if different from key, or an alias.
     */
    void add_leaf(std::string key, std::string leaf = "");

    // Properties with property mapping

    /**
     * @brief Property: A name for this event template or instance.
     *
     * This property allows setting and getting a custom name for the event,
     * which can be helpful in identifying or labeling events in complex analyses.
     */
    cproperty<std::string, event_template> name;
    /**
     * @brief Static setter for the `name` property.
     * @param[in] name Pointer to a string containing the name.
     * @param[in] ev Pointer to the `event_template` instance.
     */
    void static set_name(std::string*, event_template*);

    /**
     * @brief Property: A hash string that may identify the configuration or source of this event.
     *
     * This property stores a hash value typically used to uniquely identify the event
     * or its configuration. This can be useful for caching or cross-referencing.
     */
    cproperty<std::string, event_template> hash;
    /**
     * @brief Static setter for the `hash` property.
     * @param[in] hash Pointer to a string containing the hash.
     * @param[in] ev Pointer to the `event_template` instance.
     */
    void static set_hash(std::string*, event_template*);

    /**
     * @brief Property: The primary TTree name for this event.
     *
     * This property stores the name of the main TTree from which this event originates.
     * This is particularly relevant when working with ROOT files containing multiple TTrees.
     */
    cproperty<std::string, event_template> tree;
    /**
     * @brief Static getter for the `tree` property.
     * @param[out] name Pointer to a string to store the tree name.
     * @param[in] ev Pointer to the `event_template` instance.
     */
    void static get_tree(std::string*, event_template*);

    /**
     * @brief Property: The event weight used for normalization or scaling.
     *
     * This property stores the event weight, which is typically used for normalization,
     * cross-section adjustments, or other physics-based scalings.
     */
    cproperty<double, event_template> weight;
    /**
     * @brief Static setter for the `weight` property.
     * @param[in] val Pointer to a double containing the weight value.
     * @param[in] ev Pointer to the `event_template` instance.
     */
    void static set_weight(double*, event_template*);

    /**
     * @brief Property: The event index or entry number in the source TTree.
     *
     * This property stores the position or index of the event within the source data,
     * which is useful for tracing or identifying individual events.
     */
    cproperty<long, event_template> index;
    /**
     * @brief Static setter for the `index` property.
     * @param[in] val Pointer to a long containing the index value.
     * @param[in] ev Pointer to the `event_template` instance.
     */
    void static set_index(long*, event_template*);

    /**
     * @brief Property: List of TLeaf names (variables) to read for this event.
     *
     * This property stores the names of all variables or data fields that should be
     * extracted and made available from the raw data for this event.
     */
    cproperty<std::vector<std::string>, event_template> leaves;
    /**
     * @brief Static getter for the `leaves` property.
     * @param[out] inpt Pointer to a vector of strings to store the leaf names.
     * @param[in] ev Pointer to the `event_template` instance.
     */
    void static get_leaves(std::vector<std::string>*, event_template*);

    /**
     * @brief Internal map for storing tree name mappings or aliases.
     *
     * This map connects logical tree names (used as keys in the analysis)
     * with their actual names in the ROOT files.
     */
    std::map<std::string, std::string> m_trees = {}; ///< Map of tree name aliases.

    /**
     * @brief Internal map for storing branch name mappings or aliases.
     */
    std::map<std::string, std::string> m_branches = {}; ///< Map of branch name aliases.

    /**
     * @brief Internal map for storing leaf name mappings or aliases.
     */
    std::map<std::string, std::string> m_leaves = {}; ///< Map of leaf name aliases.

    // Particle management structures

    std::map<std::string, std::map<std::string, particle_template*>*> particle_link; ///< Map of registered particles by type and name.
    std::map<std::string, particle_template*(*)()> particle_generators; ///< Map of particle generator functions by type.

    meta* meta_data = nullptr; ///< Pointer to event metadata.
    std::string filename = ""; ///< Source filename for this event.
    event_t data; ///< Internal event data structure.
};

#endif // End of include guard for EVENT_TEMPLATE_H

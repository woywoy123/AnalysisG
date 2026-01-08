/**
 * @file container.h
 * @brief Defines the container class for managing event/graph/selection data collections.
 *
 * This file contains the declaration of the `container` class and `entry_t` structure,
 * which provide the core data management functionality for the AnalysisG framework.
 * The container holds all processed events, graphs, and selections for a data sample,
 * enabling efficient access and population of dataloaders for training.
 */

#ifndef CONTAINER_H
#define CONTAINER_H

#include <meta/meta.h>
#include <tools/tools.h>

#include <templates/graph_template.h>
#include <templates/event_template.h>
#include <templates/selection_template.h>

#include <generators/dataloader.h>

/**
 * @struct entry_t
 * @brief Data entry holding all processed data for a single event hash.
 *
 * An entry contains all the derived data from processing a single physics event,
 * including graph representations, event data, and selection results.
 */
struct entry_t {
    std::string hash = "";                           ///< Unique hash identifying this entry.
    std::vector<graph_t*>                m_data  = {}; ///< Graph tensor data for training.
    std::vector<graph_template*>         m_graph = {}; ///< Processed graph templates.
    std::vector<event_template*>         m_event = {}; ///< Source event templates.
    std::vector<selection_template*> m_selection = {}; ///< Applied selection results.

    /**
     * @brief Initializes the entry (reserves memory).
     */
    void init(); 
    
    /**
     * @brief Destroys all data in this entry, freeing memory.
     */
    void destroy(); 
    
    /**
     * @brief Checks if an event is already in this entry.
     * @param ev Pointer to the event to check.
     * @return True if the event exists in this entry.
     */
    bool has_event(event_template* ev); 
    
    /**
     * @brief Checks if a graph is already in this entry.
     * @param gr Pointer to the graph to check.
     * @return True if the graph exists in this entry.
     */
    bool has_graph(graph_template* gr); 
    
    /**
     * @brief Checks if a selection is already in this entry.
     * @param sel Pointer to the selection to check.
     * @return True if the selection exists in this entry.
     */
    bool has_selection(selection_template* sel); 
    
    /**
     * @brief Template method to destroy a vector of pointers.
     * @tparam g The type of objects in the vector.
     * @param c Pointer to the vector to destroy.
     */
    template <typename g>
    void destroy(std::vector<g*>* c){
        for (size_t x(0); x < c -> size(); ++x){
            delete (*c)[x]; 
            (*c)[x] = nullptr; 
        }
        std::vector<g*>().swap(*c); 
    }
}; 

/**
 * @class container
 * @brief Manages collections of events, graphs, and selections for a data sample.
 *
 * The container class serves as the central data store for processed physics data.
 * It holds:
 * - Metadata about the source sample
 * - Event templates with particle information
 * - Graph templates with ML-ready features
 * - Selection results for cutflow analysis
 *
 * Containers are populated during analysis execution and can be used to
 * populate dataloaders for model training.
 *
 * @section container_usage Usage
 *
 * ```cpp
 * container cont;
 * cont.add_meta_data(meta, "sample_label");
 * cont.add_event_template(my_event, "nominal");
 * cont.add_graph_template(my_graph, "particle_graph");
 * cont.add_selection_template(my_selection);
 * 
 * // Populate dataloader for training
 * dataloader dl;
 * cont.populate_dataloader(&dl);
 * ```
 */
class container: public tools
{
    public:
        /**
         * @brief Default constructor.
         */
        container();
        
        /**
         * @brief Destructor.
         * Cleans up all entries and metadata.
         */
        ~container();
        
        /**
         * @brief Associates metadata with this container.
         * @param m Pointer to the metadata object.
         * @param label Label for the data sample.
         */
        void add_meta_data(meta*, std::string); 
        
        /**
         * @brief Gets the associated metadata.
         * @return Pointer to the metadata object.
         */
        meta* get_meta_data(); 

        /**
         * @brief Adds a selection template to this container.
         * @param sel Pointer to the selection template.
         * @return True if successfully added.
         */
        bool add_selection_template(selection_template*); 
        
        /**
         * @brief Adds an event template to this container.
         * @param ev Pointer to the event template.
         * @param label Label for this event type.
         * @return True if successfully added.
         */
        bool add_event_template(event_template*, std::string label); 
        
        /**
         * @brief Adds a graph template to this container.
         * @param gr Pointer to the graph template.
         * @param label Label for this graph type.
         * @return True if successfully added.
         */
        bool add_graph_template(graph_template*, std::string label); 

        /**
         * @brief Fills a map with all selection templates.
         * @param inpt Pointer to map to fill.
         */
        void fill_selections(std::map<std::string, selection_template*>* inpt); 
        
        /**
         * @brief Gets all events with a specific label.
         * @param evts Pointer to vector to fill with events.
         * @param label The label to filter by.
         */
        void get_events(std::vector<event_template*>*, std::string label); 
        
        /**
         * @brief Populates a dataloader with this container's data.
         * @param dl Pointer to the dataloader to populate.
         */
        void populate_dataloader(dataloader* dl);
        
        /**
         * @brief Compiles all entries in this container.
         * @param len Pointer to store the total length.
         * @param threadIdx Thread index for parallel processing.
         */
        void compile(size_t* len, int threadIdx); 
        
        /**
         * @brief Returns the number of entries in this container.
         * @return Number of entries.
         */
        size_t len(); 
        
        /**
         * @brief Adds or retrieves an entry by hash.
         * @param hash The unique hash for this entry.
         * @return Pointer to the entry.
         */
        entry_t* add_entry(std::string hash); 

        meta*        meta_data   = nullptr; ///< Associated metadata.
        std::string* filename    = nullptr; ///< Source filename.
        std::string* output_path = nullptr; ///< Output path for results.
        std::string     label    = "";      ///< Container label.

        std::map<std::string, entry_t> random_access; ///< Map of entries by hash.
        std::map<std::string, selection_template*>* merged = nullptr; ///< Merged selections.
}; 

#endif

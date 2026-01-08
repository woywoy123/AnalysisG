/**
 * @file dataloader.h
 * @brief Defines the dataloader class for managing graph data batching and training.
 *
 * This file contains the declaration of the `dataloader` class, which provides
 * functionality for batch generation, k-fold cross-validation, data shuffling,
 * and GPU memory management during model training.
 */

#ifndef DATALOADER_GENERATOR_H
#define DATALOADER_GENERATOR_H

#ifdef PYC_CUDA
#include <cuda.h>
#include <c10/cuda/CUDACachingAllocator.h>
#define _server true
#endif

#ifndef _server
#define _server false
#endif

#include <tools/tools.h>
#include <structs/property.h>
#include <structs/settings.h>
#include <notification/notification.h>
#include <templates/graph_template.h>

#include <map>
#include <random>
#include <algorithm>

class analysis; 
class model_template; 
struct model_report; 

/**
 * @class dataloader
 * @brief Manages graph data loading, batching, and dataset splitting for ML training.
 *
 * The dataloader class provides comprehensive functionality for:
 * - Creating batches from graph data for model training
 * - K-fold cross-validation dataset splitting
 * - Test/train set generation with configurable splits
 * - Data shuffling and randomization
 * - GPU memory management and CUDA server functionality
 * - Dataset persistence (save/restore)
 *
 * @section dataloader_usage Basic Usage
 *
 * ```cpp
 * dataloader dl;
 * 
 * // Add graphs to the dataloader
 * for (auto& graph : graphs) {
 *     dl.extract_data(graph);
 * }
 * 
 * // Generate train/test split
 * dl.generate_test_set(20.0);  // 20% test set
 * 
 * // Generate k-fold splits
 * dl.generate_kfold_set(5);  // 5-fold CV
 * 
 * // Get training data for fold 0
 * auto* train_set = dl.get_k_train_set(0);
 * 
 * // Build batches for training
 * auto* batch = dl.build_batch(train_set, model, report);
 * ```
 *
 * @section dataloader_kfold K-Fold Cross-Validation
 *
 * The dataloader supports k-fold cross-validation:
 * - `generate_kfold_set(k)`: Creates k training/validation splits
 * - `get_k_train_set(i)`: Gets training set for fold i
 * - `get_k_validation_set(i)`: Gets validation set for fold i
 *
 * @section dataloader_batching Batching
 *
 * Batching combines multiple graphs into a single batch by:
 * - Concatenating node features with batch indexing
 * - Adjusting edge indices for the combined graph
 * - Stacking graph-level features
 *
 * @see graph_t
 * @see model_template
 */
class dataloader: 
    public notification, 
    public tools
{
    public:
        /**
         * @brief Default constructor.
         */
        dataloader();
        
        /**
         * @brief Destructor.
         * Cleans up all cached batches and dataset splits.
         */
        ~dataloader();

        /**
         * @brief Gets the training set for k-fold cross-validation.
         * @param k Fold index (0 to k-1).
         * @return Pointer to vector of training graphs for this fold.
         */
        std::vector<graph_t*>* get_k_train_set(int k); 
        
        /**
         * @brief Gets the validation set for k-fold cross-validation.
         * @param k Fold index (0 to k-1).
         * @return Pointer to vector of validation graphs for this fold.
         */
        std::vector<graph_t*>* get_k_validation_set(int k); 
        
        /**
         * @brief Gets the test set.
         * @return Pointer to vector of test graphs.
         */
        std::vector<graph_t*>* get_test_set(); 
        
        /**
         * @brief Builds a batch from a set of graphs.
         * @param data Pointer to vector of graphs to batch.
         * @param mdl Pointer to model (for device info).
         * @param rep Pointer to model report (for statistics).
         * @return Pointer to vector containing the batched graphs.
         */
        std::vector<graph_t*>* build_batch(std::vector<graph_t*>* data, model_template* mdl, model_report* rep); 
        
        /**
         * @brief Safely deletes a vector of graph data.
         * @param data Pointer to vector to delete.
         */
        static void safe_delete(std::vector<graph_t*>* data); 

        /**
         * @brief Gets all data organized by sample for inference.
         * @return Pointer to map of sample names to graph vectors.
         */
        std::map<std::string, std::vector<graph_t*>>* get_inference(); 

        /**
         * @brief Generates train/test split.
         * @param percentage Percentage of data to use for testing (0-100).
         */
        void generate_test_set(float percentage = 50); 
        
        /**
         * @brief Generates k-fold cross-validation splits.
         * @param k Number of folds.
         */
        void generate_kfold_set(int k); 
        
        /**
         * @brief Saves the dataset split configuration to disk.
         * @param path Path to save the configuration.
         */
        void dump_dataset(std::string path); 
        
        /**
         * @brief Restores a dataset split configuration from disk.
         * @param path Path to load the configuration from.
         * @return True if restore succeeded.
         */
        bool restore_dataset(std::string path); 

        /**
         * @brief Gets a random sample of graphs.
         * @param num Number of graphs to sample.
         * @return Vector of randomly selected graphs.
         */
        std::vector<graph_t*> get_random(int num = 5); 
        
        /**
         * @brief Extracts and stores a graph's data.
         * @param gr Pointer to graph to extract.
         */
        void extract_data(graph_t* gr); 
        
        /**
         * @brief Transfers all data to a specific device.
         * @param op TensorOptions specifying target device.
         * @param num_events Pointer to total event count (for progress).
         * @param prg_events Pointer to progress counter.
         */
        void datatransfer(torch::TensorOptions* op, size_t* num_events = nullptr, size_t* prg_events = nullptr);
        
        /**
         * @brief Transfers data to multiple devices.
         * @param ops Map of device indices to TensorOptions.
         */
        void datatransfer(std::map<int, torch::TensorOptions*>* ops);
        
        /**
         * @brief Saves all graphs to HDF5 files.
         * @param path Output directory path.
         * @param threads Number of threads for parallel writing.
         * @return True if successful.
         */
        bool dump_graphs(std::string path = "./", int threads = 10); 

        /**
         * @brief Restores graphs from a list of HDF5 files.
         * @param paths Vector of file paths.
         * @param threads Number of threads for parallel reading.
         */
        void restore_graphs(std::vector<std::string> paths, int threads); 
        
        /**
         * @brief Restores graphs from a directory.
         * @param paths Directory path or glob pattern.
         * @param threads Number of threads for parallel reading.
         */
        void restore_graphs(std::string paths, int threads); 
        
        /**
         * @brief Starts the CUDA memory management server.
         * Monitors and manages GPU memory during training.
         */
        void start_cuda_server(); 
    
    private:
        friend class analysis;
        settings_t* setting = nullptr; 
        std::thread* cuda_mem = nullptr; 

        void cuda_memory_server(); 

        void clean_data_elements(
                std::map<std::string, int>** data_map, 
                std::vector<std::map<std::string, int>*>* loader_map
        ); 

        void shuffle(std::vector<int>* idx); 
        void shuffle(std::vector<graph_t*>* idx); 
        std::map<std::string, graph_t*>* restore_graphs_(std::vector<std::string>, int threads); 

        std::map<int, std::vector<graph_t*>*> batched_cache = {}; 
        std::map<int, std::vector<graph_t*>*> gr_k_fold_training = {}; 
        std::map<int, std::vector<graph_t*>*> gr_k_fold_validation = {}; 


        std::map<int, std::vector<int>*> k_fold_training = {}; 
        std::map<int, std::vector<int>*> k_fold_validation = {}; 

        std::vector<int>* test_set  = nullptr; 
        std::vector<int>* train_set = nullptr; 
        std::vector<int>* data_index = nullptr; 

        std::vector<std::map<std::string, int>*> truth_map_graph = {}; 
        std::vector<std::map<std::string, int>*> truth_map_node  = {}; 
        std::vector<std::map<std::string, int>*> truth_map_edge  = {}; 

        std::vector<std::map<std::string, int>*> data_map_graph = {}; 
        std::vector<std::map<std::string, int>*> data_map_node  = {}; 
        std::vector<std::map<std::string, int>*> data_map_edge  = {}; 

        std::map<std::string, int> hash_map = {}; 
        std::vector<graph_t*>*  gr_test = nullptr; 
        std::vector<graph_t*>* data_set = nullptr; 

        std::default_random_engine rnd{}; 
}; 

#endif

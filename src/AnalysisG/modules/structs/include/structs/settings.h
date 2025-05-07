/**
 * @file settings.h
 * @brief Defines the `settings_t` structure for global configuration of the AnalysisG framework.
 *
 * This file contains the declaration of the `settings_t` structure, which serves as a central
 * configuration container for the AnalysisG framework. It holds various settings related to
 * file paths, machine learning parameters, visualization options, and runtime behavior
 * that affect the overall operation of the analysis pipeline.
 */

#ifndef SETTINGS_STRUCTS_H ///< Include guard start for SETTINGS_STRUCTS_H.
#define SETTINGS_STRUCTS_H ///< Definition of the include guard.

#include <string> ///< Includes the standard string library for string handling.
#include <vector> ///< Includes the standard vector container for sequence storage.

/**
 * @struct settings_t
 * @brief Contains all configuration settings for an analysis session.
 *
 * The settings_t structure is used throughout the AnalysisG framework to configure
 * various aspects of the analysis pipeline, including paths for input/output,
 * machine learning parameters, visualization settings, and runtime behavior options.
 */
struct settings_t {
    /**
     * @brief Path where output files and directories will be created.
     * Default is "./ProjectName".
     */
    std::string output_path = "./ProjectName";
    
    /**
     * @brief Name for this analysis run or session, used for organizing outputs.
     * Default is empty string.
     */
    std::string run_name = "";
    
    /**
     * @brief Name for Sum-Of-Weights reference, used in normalization.
     * Default is empty string.
     */
    std::string sow_name = "";
    
    /**
     * @brief Path for caching metadata to avoid repeated retrieval.
     * Default is current directory ("./").
     */
    std::string metacache_path = "./";
    
    /**
     * @brief Whether to fetch metadata from external sources (e.g., AMI database).
     * Default is false.
     */
    bool fetch_meta = false;
    
    /**
     * @brief Whether to apply pre-tagging to events before full processing.
     * Default is false.
     */
    bool pretagevents = false;

    // Machine learning settings
    
    /**
     * @brief Number of training epochs for machine learning models.
     * Default is 10.
     */
    int epochs = 10;
    
    /**
     * @brief Number of folds for k-fold cross-validation.
     * Default is 10.
     */
    int kfolds = 10;
    
    /**
     * @brief Batch size for model training.
     * Default is 1.
     */
    int batch_size = 1;
    
    /**
     * @brief Specific fold indices to use if not using all folds.
     * Default is empty (use all folds).
     */
    std::vector<int> kfold = {};

    /**
     * @brief Number of example events/objects to generate or display.
     * Default is 3.
     */
    int num_examples = 3;
    
    /**
     * @brief Percentage of data to use for training (vs validation/testing).
     * Default is 50%.
     */
    float train_size = 50;
   
    /**
     * @brief Whether to perform model training.
     * Default is true.
     */
    bool training = true;
    
    /**
     * @brief Whether to perform validation during training.
     * Default is true.
     */
    bool validation = true;
    
    /**
     * @brief Whether to evaluate models after training.
     * Default is true.
     */
    bool evaluation = true;
    
    /**
     * @brief Whether to continue training from previous checkpoints if available.
     * Default is true.
     */
    bool continue_training = true;

    /**
     * @brief Path or identifier for the training dataset.
     * Default is empty string.
     */
    std::string training_dataset = "";
    
    /**
     * @brief Path for caching graph structures to avoid recomputation.
     * Default is empty string.
     */
    std::string graph_cache = "";

    // Plotting configuration variables
    
    /**
     * @brief Name of the variable representing transverse momentum for plotting.
     * Default is "pt".
     */
    std::string var_pt = "pt";
    
    /**
     * @brief Name of the variable representing pseudorapidity for plotting.
     * Default is "eta".
     */
    std::string var_eta = "eta";
    
    /**
     * @brief Name of the variable representing azimuthal angle for plotting.
     * Default is "phi".
     */
    std::string var_phi = "phi";
    
    /**
     * @brief Name of the variable representing energy for plotting.
     * Default is "energy".
     */
    std::string var_energy = "energy";
    
    /**
     * @brief Target variables or attributes for modeling or plotting.
     * Default is empty.
     */
    std::vector<std::string> targets = {};
    
    /**
     * @brief Number of bins for histograms.
     * Default is 400.
     */
    int nbins = 400;
    
    /**
     * @brief Refresh rate for progress displays, in iterations.
     * Default is 10.
     */
    int refresh = 10;
    
    /**
     * @brief Maximum range for histograms or plots.
     * Default is 400.
     */
    int max_range = 400;

    /**
     * @brief Number of threads for parallel processing.
     * Default is 10.
     */
    int threads = 10;
    
    /**
     * @brief Whether to enable additional debug output.
     * Default is false.
     */
    bool debug_mode = false;
    
    /**
     * @brief Whether to build and use a cache for intermediate results.
     * Default is false.
     */
    bool build_cache = false;
    
    /**
     * @brief Whether to store selection results in ROOT format.
     * Default is false.
     */
    bool selection_root = false;
}; 

#endif // SETTINGS_STRUCTS_H ///< End of include guard.

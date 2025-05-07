/**
 * @file model.h
 * @brief Defines data structures for model configuration in the AnalysisG framework.
 *
 * This file contains the definition of the model_settings_t structure, which holds
 * configuration parameters for machine learning models in the AnalysisG framework.
 * It includes settings for optimizers, model identification, checkpoint paths, and
 * input/output feature mappings.
 */

#ifndef STRUCTS_MODEL_H ///< Start of include guard for STRUCTS_MODEL_H.
#define STRUCTS_MODEL_H ///< Definition of STRUCTS_MODEL_H to signify the header has been included.

#include <structs/enums.h> ///< Include enumerations used for optimizer types.
#include <string> ///< Include standard string library for string manipulation.
#include <vector> ///< Include standard vector container for sequence storage.
#include <map> ///< Include standard map container for key-value associations.

/**
 * @struct model_settings_t
 * @brief Structure for storing model configuration settings.
 *
 * This structure holds various parameters that define how a machine learning model
 * should be configured and trained. It includes settings for the optimizer, paths
 * for checkpoints, device selection, and mappings for input and output features.
 */
struct model_settings_t {
    opt_enum    e_optim;   ///< Optimizer type as an enumerated value.
    std::string s_optim;   ///< Optimizer type as a string representation.

    std::string weight_name;  ///< Name of weight file to use for the model.
    std::string tree_name;    ///< Name of the tree to use for training/inference.

    std::string model_name;            ///< Unique identifier for the model.
    std::string model_device;          ///< Device to use for model execution (e.g., "cpu", "cuda:0").
    std::string model_checkpoint_path; ///< Path where model checkpoints will be saved.

    bool binary_classification = true; ///< Flag indicating if the model performs binary classification. Default is true.
    
    // Feature mappings for graph neural networks
    std::map<std::string, std::string> i_graph_features; ///< Mapping of input graph feature names to their actual source names.
    std::map<std::string, std::string> i_node_features;  ///< Mapping of input node feature names to their actual source names.
    std::map<std::string, std::string> i_edge_features;  ///< Mapping of input edge feature names to their actual source names.
    
    std::map<std::string, std::string> o_graph_features; ///< Mapping of output graph feature names to their target names.
    std::map<std::string, std::string> o_node_features;  ///< Mapping of output node feature names to their target names.
    std::map<std::string, std::string> o_edge_features;  ///< Mapping of output edge feature names to their target names.
    
    // Other model configuration settings
    std::map<std::string, int> model_integers;    ///< Integer parameters for the model (e.g., layer sizes, feature counts).
    std::map<std::string, float> model_floats;    ///< Float parameters for the model (e.g., learning rates, regularization factors).
    std::map<std::string, std::string> model_strings; ///< String parameters for the model (e.g., activation functions, layer types).
    
    /**
     * @brief Validates the model settings to ensure all required parameters are set.
     * @return True if the settings are valid, false otherwise.
     */
    bool validate() const {
        return !model_name.empty() && !model_checkpoint_path.empty();
    }
}; 

#endif // STRUCTS_MODEL_H ///< End of include guard for STRUCTS_MODEL_H.

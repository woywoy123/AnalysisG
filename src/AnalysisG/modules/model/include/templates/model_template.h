/**
 * @file model_template.h
 * @brief Defines the base template class for machine learning models in the AnalysisG framework.
 *
 * This file contains the declaration of the `model_template` class, which serves as a base
 * template for creating machine learning models. It defines a standard interface for model
 * initialization, training, and inference, particularly for graph-based neural networks.
 * The class handles model parameters, device placement, feature assignment, and forward propagation.
 */

#ifndef MODEL_TEMPLATE_H ///< Start of include guard for MODEL_TEMPLATE_H to prevent multiple inclusions.
#define MODEL_TEMPLATE_H ///< Definition of MODEL_TEMPLATE_H to signify the header has been included.

#include <torch/torch.h> ///< Includes PyTorch C++ frontend for deep learning.
#include <string> ///< Includes the standard string library for string manipulation.
#include <vector> ///< Includes the standard vector container for sequence storage.
#include <map> ///< Includes the standard map container for key-value associations.

#include <notification/notification.h> ///< Includes the notification class for logging.
#include <tools/tools.h> ///< Includes the tools class for utility functions.
#include <structs/property.h> ///< Includes the property template for custom property management.
#include <structs/graph.h> ///< Includes the graph_t structure for graph data representation.
#include <structs/model.h> ///< Includes the model_settings_t structure for model configuration.
#include <lossfx/lossfx.h> ///< Includes the lossfx class for defining loss functions.

/**
 * @enum mlp_init
 * @brief Enumeration of initialization methods for neural network weights.
 */
enum mlp_init {
    xavier = 0, ///< Xavier/Glorot initialization, balancing signal variance across layers.
    kaiming = 1, ///< Kaiming/He initialization, particularly good for ReLU activations.
    normal = 2,  ///< Standard normal distribution initialization.
    uniform = 3  ///< Uniform distribution initialization.
};

/**
 * @class model_template
 * @brief Base template class for machine learning models.
 *
 * Inherits from `tools` and `notification` to provide utility functions and logging capabilities.
 * This class serves as a foundation for implementing various neural network architectures,
 * with particular focus on graph neural networks. It provides mechanisms for managing
 * model parameters, feature assignment, forward propagation, and more.
 */
class model_template: 
    public tools,
    public notification
{
public:
    /**
     * @brief Constructor for the `model_template` class.
     * Initializes a new model template with default settings.
     */
    model_template();
    
    /**
     * @brief Virtual destructor for the `model_template` class.
     * Ensures proper cleanup of resources, including module containers and outputs.
     */
    virtual ~model_template();
    
    /**
     * @brief Creates a clone of the model template.
     * @return A pointer to a new model_template instance that is a copy of this one.
     */
    virtual model_template* clone();
    
    /**
     * @brief Registers a PyTorch sequential module with the model.
     * @param data Pointer to the sequential module to register.
     * 
     * Adds the module to the model's collection and places it on the appropriate device.
     */
    void register_module(torch::nn::Sequential* data);
    
    /**
     * @brief Registers a PyTorch sequential module with the model and initializes its weights.
     * @param data Pointer to the sequential module to register.
     * @param method The initialization method to use for the module's weights.
     * 
     * Adds the module to the model's collection, places it on the appropriate device,
     * and initializes its weights according to the specified method.
     */
    void register_module(torch::nn::Sequential* data, mlp_init method);
    
    /**
     * @brief Virtual forward pass method for a single graph.
     * @param data Pointer to the graph_t structure containing input data.
     * 
     * This method should be overridden by derived classes to implement the forward pass logic.
     */
    virtual void forward(graph_t* data);
    
    /**
     * @brief Forward pass for a single graph with training mode flag.
     * @param data Pointer to the graph_t structure containing input data.
     * @param train Flag indicating if the model is in training mode (true) or evaluation mode (false).
     *
     * Executes the model's forward pass on a single graph, optionally in training mode.
     */
    virtual void forward(graph_t* data, bool train);
    
    /**
     * @brief Forward pass for multiple graphs with training mode flag.
     * @param data Vector of pointers to graph_t structures containing input data.
     * @param train Flag indicating if the model is in training mode (true) or evaluation mode (false).
     *
     * Executes the model's forward pass on multiple graphs, optionally in training mode.
     */
    virtual void forward(std::vector<graph_t*> data, bool train);
    
    /**
     * @brief Assigns features from a graph to tensors based on feature name and type.
     * @param inpt Feature name to assign.
     * @param type Type of graph component (node, edge, or graph).
     * @param data Pointer to the graph_t structure containing the features.
     * @return Pointer to the resulting tensor containing the assigned features.
     */
    torch::Tensor* assign_features(std::string inpt, graph_enum type, graph_t* data);
    
    /**
     * @brief Assigns features from multiple graphs to tensors based on feature name and type.
     * @param inpt Feature name to assign.
     * @param type Type of graph component (node, edge, or graph).
     * @param data Pointer to a vector of graph_t structures containing the features.
     * @return Pointer to the resulting tensor containing the assigned features.
     */
    torch::Tensor* assign_features(std::string inpt, graph_enum type, std::vector<graph_t*>* data);
    
    /**
     * @brief Clears all output tensors, freeing memory.
     * Used to reset the model's output state between forward passes.
     */
    void flush_outputs();

    // Property declarations for various model settings and features
    cproperty<std::string, model_template> name; ///< Property: The name of this model.
    /** @brief Static setter for the `name` property.
     *  @param[in] name Pointer to a string containing the name.
     *  @param[in] model Pointer to the `model_template` instance. */
    void static set_name(std::string* name, model_template* model);
    /** @brief Static getter for the `name` property.
     *  @param[out] name Pointer to a string to store the name.
     *  @param[in] model Pointer to the `model_template` instance. */
    void static get_name(std::string* name, model_template* model);

    cproperty<std::string, model_template> device; ///< Property: The device this model runs on (e.g., "cpu", "cuda:0").
    /** @brief Static setter for the `device` property.
     *  @param[in] dev Pointer to a string containing the device name.
     *  @param[in] model Pointer to the `model_template` instance. */
    void static set_device(std::string* dev, model_template* model);

    cproperty<int, model_template> device_index; ///< Property: The numerical index of the device this model runs on.
    /** @brief Static setter for the `device_index` property.
     *  @param[in] dev_idx Pointer to an integer containing the device index.
     *  @param[in] model Pointer to the `model_template` instance. */
    void static set_dev_index(int* dev_idx, model_template* model);
    /** @brief Static getter for the `device_index` property.
     *  @param[out] dev_idx Pointer to an integer to store the device index.
     *  @param[in] model Pointer to the `model_template` instance. */
    void static get_dev_index(int* dev_idx, model_template* model);

    // Input feature properties
    cproperty<std::vector<std::string>, std::map<std::string, torch::Tensor*>> i_graph; ///< Property: Graph-level input features.
    cproperty<std::vector<std::string>, std::map<std::string, torch::Tensor*>> i_node;  ///< Property: Node-level input features.
    cproperty<std::vector<std::string>, std::map<std::string, torch::Tensor*>> i_edge;  ///< Property: Edge-level input features.

    // Output feature properties
    cproperty<std::vector<std::string>, std::map<std::string, torch::Tensor*>> o_graph; ///< Property: Graph-level output features.
    cproperty<std::vector<std::string>, std::map<std::string, torch::Tensor*>> o_node;  ///< Property: Node-level output features.
    cproperty<std::vector<std::string>, std::map<std::string, torch::Tensor*>> o_edge;  ///< Property: Edge-level output features.

    /**
     * @brief Sets input features for a specific feature map.
     * @param inpt Pointer to a vector of strings containing feature names.
     * @param in_fx Pointer to the feature map to set.
     */
    void static set_input_features(std::vector<std::string>* inpt, std::map<std::string, torch::Tensor*>* in_fx);
    
    /**
     * @brief Sets output features for a specific feature map.
     * @param inpt Pointer to a vector of strings containing feature names.
     * @param in_fx Pointer to the feature map to set.
     */
    void static set_output_features(std::vector<std::string>* inpt, std::map<std::string, torch::Tensor*>* in_fx);

    // Storage for feature maps
    std::map<std::string, torch::Tensor*> m_i_graph; ///< Map of graph-level input features to tensors.
    std::map<std::string, torch::Tensor*> m_i_node;  ///< Map of node-level input features to tensors.
    std::map<std::string, torch::Tensor*> m_i_edge;  ///< Map of edge-level input features to tensors.
    
    std::map<std::string, torch::Tensor*> m_o_graph; ///< Map of graph-level output features to tensors.
    std::map<std::string, torch::Tensor*> m_o_node; ///< Map of node-level output features to tensors.
    std::map<std::string, torch::Tensor*> m_o_edge; ///< Map of edge-level output features to tensors.

    torch::TensorOptions* m_option = nullptr; ///< Pointer to tensor options for configuring tensor creation.
    std::vector<torch::nn::Sequential*> m_data; ///< Vector of pointers to sequential modules in the model.
    lossfx* m_loss = nullptr; ///< Pointer to the loss function manager.

    std::string model_checkpoint_path; ///< Path for saving/loading model checkpoints.
};

#endif ///< End of include guard for MODEL_TEMPLATE_H.

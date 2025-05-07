/**
 * @file model_template.pxd
 * @brief Provides type definitions and class declarations for model templates in the AnalysisG framework.
 *
 * This file defines the structure and behavior of model templates, including methods for managing
 * models, training configurations, and evaluation processes.
 */

# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "<templates/model_template.h>" nogil:
    cdef cppclass model_template:

        model_template() except+ nogil

        map[string, string] o_graph
        map[string, string] o_node
        map[string, string] o_edge

        vector[string] i_graph
        vector[string] i_node
        vector[string] i_edge

        string device
        string name
        string weight_name
        string tree_name
        string model_checkpoint_path

/**
 * @namespace ModelTemplate
 * @brief Contains the ModelTemplate class and related utilities for managing models.
 */

/**
 * @class ModelTemplate
 * @brief Represents a machine learning model used in the analysis.
 *
 * This class provides methods for setting model parameters, training the model, and evaluating its performance.
 */
cdef class ModelTemplate:

    cdef bool rename
    cdef model_template* nn_ptr;
    cdef dict conv(self, map[string, string]*)
    cdef map[string, string] cond(self, dict inpt)

    cdef dict parameters_ ///< Dictionary of parameters associated with the model.
    cdef dict results_ ///< Dictionary of results computed during model evaluation.

    /**
     * @brief Sets a parameter for the model.
     *
     * @param key The key of the parameter.
     * @param value The value of the parameter.
     */
    void set_parameter(string key, string value)

    /**
     * @brief Retrieves a parameter from the model.
     *
     * @param key The key of the parameter.
     * @return The value of the parameter.
     */
    string get_parameter(string key)

    /**
     * @brief Trains the model using the provided data.
     *
     * @param data The data used for training the model.
     */
    void train_model(dict data)

    /**
     * @brief Evaluates the model's performance.
     *
     * @param data The data used for evaluation.
     * @return A dictionary containing evaluation results.
     */
    dict evaluate_model(dict data)

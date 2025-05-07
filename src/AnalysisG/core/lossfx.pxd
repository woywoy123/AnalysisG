/**
 * @file lossfx.pxd
 * @brief Provides type definitions and class declarations for loss functions in the AnalysisG framework.
 *
 * This file defines the structure and behavior of loss functions, including methods for configuring
 * and computing loss values during model training.
 */

# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
cimport cython.operator

cdef extern from "<structs/optimizer.h>" nogil:

    cdef cppclass optimizer_params_t:

        void operator()() except+ nogil
        string optimizer

        double lr
        double lr_decay
        double weight_decay
        double initial_accumulator_value
        double eps
        double tolerance_grad
        double tolerance_change
        double alpha
        double momentum
        double dampening

        bool amsgrad
        bool centered
        bool nesterov

        int max_iter
        int max_eval
        int history_size

        vector[float] beta_hack

cdef class OptimizerConfig:

    cdef optimizer_params_t* params

/**
 * @namespace LossFunctions
 * @brief Contains the LossFunctions class and related utilities for managing loss computations.
 */

/**
 * @class LossFunctions
 * @brief Represents a collection of loss functions used in model training.
 *
 * This class provides methods for setting loss parameters, computing loss values, and retrieving results.
 */
cdef class LossFunctions:
    cdef dict parameters_ ///< Dictionary of parameters associated with the loss function.
    cdef float loss_value_ ///< The computed loss value.

    /**
     * @brief Sets a parameter for the loss function.
     *
     * @param key The key of the parameter.
     * @param value The value of the parameter.
     */
    void set_parameter(string key, string value)

    /**
     * @brief Retrieves a parameter from the loss function.
     *
     * @param key The key of the parameter.
     * @return The value of the parameter.
     */
    string get_parameter(string key)

    /**
     * @brief Computes the loss value based on the provided data.
     *
     * @param predictions The model predictions.
     * @param targets The ground truth targets.
     */
    void compute_loss(dict predictions, dict targets)

    /**
     * @brief Retrieves the computed loss value.
     *
     * @return The computed loss value.
     */
    float get_loss_value()

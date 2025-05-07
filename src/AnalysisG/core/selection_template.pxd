/**
 * @file selection_template.pxd
 * @brief Provides type definitions and class declarations for selection templates in the AnalysisG framework.
 *
 * This file defines the structure and behavior of selection templates, including methods for managing
 * selection criteria, applying filters, and retrieving selected data.
 */

# distutils: language=c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string

from AnalysisG.core.structs cimport meta_t
from AnalysisG.core.event_template cimport event_template

cdef extern from "<templates/selection_template.h>" nogil:
    cdef cppclass selection_template:
        selection_template() except+ nogil

        string name
        string hash
        string tree
        double index

        selection_template* build(event_template*) except+ nogil
        bool selection(event_template*) except+ nogil
        bool strategy(event_template*) except+ nogil
        void merge(selection_template*) except+ nogil

        vector[map[string, float]] reverse_hash(vector[string]* hashes) except+ nogil
        bool CompileEvent() except+ nogil

        bool operator == (selection_template& p) except+ nogil
        map[string, map[string, float]] passed_weights
        map[string, meta_t] matched_meta

/**
 * @namespace SelectionTemplate
 * @brief Contains the SelectionTemplate class and related utilities for managing selection criteria.
 */

/**
 * @class SelectionTemplate
 * @brief Represents a selection criterion used for filtering data.
 *
 * This class provides methods for setting selection parameters, applying filters, and retrieving results.
 */
cdef class SelectionTemplate:
    cdef selection_template* ptr
    cdef void transform_dict_keys(self)
    cdef public dict root_leaves
    cdef dict parameters_ ///< Dictionary of parameters associated with the selection.
    cdef list filters_ ///< List of filters applied to the data.

    /**
     * @brief Sets a parameter for the selection.
     *
     * @param key The key of the parameter.
     * @param value The value of the parameter.
     */
    void set_parameter(string key, string value)

    /**
     * @brief Retrieves a parameter from the selection.
     *
     * @param key The key of the parameter.
     * @return The value of the parameter.
     */
    string get_parameter(string key)

    /**
     * @brief Applies the selection criteria to the data.
     *
     * @param data The data to which the selection criteria are applied.
     */
    void apply_selection(dict data)

    /**
     * @brief Retrieves the results of the selection.
     *
     * @return A list containing the selected data.
     */
    list get_results()

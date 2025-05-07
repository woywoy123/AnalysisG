/**
 * @file metric_template.pxd
 * @brief Provides type definitions and class declarations for metric templates in the AnalysisG framework.
 *
 * This file defines the structure and behavior of metric templates, including methods for managing
 * metrics, evaluating models, and computing performance metrics.
 */

# distutils: language=c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.tools cimport *
from AnalysisG.core.io cimport *
from cython.operator cimport dereference as deref

cdef extern from "<templates/metric_template.h>" nogil:
    cdef cppclass metric_template(tools, notification):
        metric_template() except+ nogil
        string name
        map[string, string] run_names
        vector[string] variables

cdef inline bool finder(string* fname, vector[string]* kfolds, vector[string]* epochs) nogil:
    cdef tools tl
    cdef string ix
    cdef bool found_k = False
    cdef bool found_e = False
    for ix in deref(kfolds):
        if not tl.ends_with(fname, string(b"kfold-") + ix + string(b".root")): continue
        found_k = True; break

    for ix in deref(epochs):
        if not tl.has_string(fname, string(b"epoch-") + ix + string(b"/")): continue
        found_e = True; break
    return (kfolds.size() == 0 or found_k)*(epochs.size() == 0 or found_e)

/**
 * @namespace MetricTemplate
 * @brief Contains the MetricTemplate class and related utilities for managing metrics.
 */

/**
 * @class MetricTemplate
 * @brief Represents a metric used for evaluating models.
 *
 * This class provides methods for setting metric parameters, computing metrics, and retrieving results.
 */
cdef class MetricTemplate(Tools):
    cdef metric_template* mtx
    cdef public dict root_leaves
    cdef public dict root_fx
    cdef dict parameters_ ///< Dictionary of parameters associated with the metric.
    cdef dict results_ ///< Dictionary of results computed by the metric.

    /**
     * @brief Sets a parameter for the metric.
     *
     * @param key The key of the parameter.
     * @param value The value of the parameter.
     */
    void set_parameter(string key, string value)

    /**
     * @brief Retrieves a parameter from the metric.
     *
     * @param key The key of the parameter.
     * @return The value of the parameter.
     */
    string get_parameter(string key)

    /**
     * @brief Computes the metric based on the provided data.
     *
     * @param data The data used for computing the metric.
     */
    void compute_metric(dict data)

    /**
     * @brief Retrieves the results of the metric computation.
     *
     * @return A dictionary containing the results.
     */
    dict get_results()

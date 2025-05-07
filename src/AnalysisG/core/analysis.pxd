/**
 * @file analysis.pxd
 * @brief Provides type definitions and class declarations for the Analysis class in the AnalysisG framework.
 *
 * This file defines the structure and behavior of the Analysis class, including methods for adding samples,
 * events, graphs, selections, and metrics, as well as managing analysis settings and progress.
 */

# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.notification cimport notification
from AnalysisG.core.event_template cimport event_template
from AnalysisG.core.graph_template cimport graph_template
from AnalysisG.core.selection_template cimport selection_template
from AnalysisG.core.metric_template cimport metric_template
from AnalysisG.core.model_template cimport model_template

from AnalysisG.core.lossfx cimport optimizer_params_t
from AnalysisG.core.structs cimport settings_t
from AnalysisG.core.meta cimport *

cdef extern from "<AnalysisG/analysis.h>" nogil:

    cdef cppclass analysis(notification):
        analysis() except+ nogil

        void add_samples(string path, string label) except+ nogil
        void add_selection_template(selection_template* ev) except+ nogil
        void add_event_template(event_template* ev, string label) except+ nogil
        void add_graph_template(graph_template* ev, string label) except+ nogil
        void add_metric_template(metric_template* ev, model_template* mdl) except+ nogil

        void add_model(model_template* model, optimizer_params_t* op, string run_name) except+ nogil
        void add_model(model_template* model, string run_name) except+ nogil

        void start() except+ nogil
        void attach_threads() except+ nogil

        map[string, vector[float]] progress() except+ nogil
        map[string, string] progress_mode() except+ nogil
        map[string, string] progress_report() except+ nogil
        map[string, bool] is_complete() except+

        # settings
        map[string, meta*] meta_data
        settings_t m_settings

/**
 * @namespace Analysis
 * @brief Contains the Analysis class and related utilities for managing data processing.
 */

/**
 * @class Analysis
 * @brief Manages data processing, including samples, events, graphs, and metrics.
 *
 * This class provides methods for configuring and executing analysis workflows, as well as tracking progress.
 */
cdef class Analysis:
    cdef list selections_ ///< List of selection templates added to the analysis.
    cdef list graphs_ ///< List of graph templates added to the analysis.
    cdef list events_ ///< List of event templates added to the analysis.
    cdef list models_ ///< List of models added to the analysis.
    cdef list optim_ ///< List of optimizers associated with the models.
    cdef dict meta_ ///< Dictionary for storing metadata associated with the analysis.
    cdef analysis* ana ///< Pointer to the underlying C++ analysis object.

    /**
     * @brief Adds a graph template to the analysis.
     *
     * @param ev Pointer to the graph template.
     * @param label Label for the graph template.
     */
    void add_graph_template(graph_template* ev, string label) except+ nogil

    /**
     * @brief Adds a metric template to the analysis.
     *
     * @param ev Pointer to the metric template.
     * @param mdl Pointer to the model template associated with the metric.
     */
    void add_metric_template(metric_template* ev, model_template* mdl) except+ nogil

    /**
     * @brief Adds a model to the analysis.
     *
     * @param model Pointer to the model template.
     * @param op Pointer to the optimizer parameters.
     * @param run_name Name of the run associated with the model.
     */
    void add_model(model_template* model, optimizer_params_t* op, string run_name) except+ nogil

    /**
     * @brief Adds a model for inference to the analysis.
     *
     * @param model Pointer to the model template.
     * @param run_name Name of the run associated with the model.
     */
    void add_model(model_template* model, string run_name) except+ nogil

    /**
     * @brief Starts the analysis process.
     */
    void start() except+ nogil

    /**
     * @brief Attaches threads for parallel processing.
     */
    void attach_threads() except+ nogil

    /**
     * @brief Retrieves the progress of the analysis.
     *
     * @return A map containing progress information for each component.
     */
    map[string, vector[float]] progress() except+ nogil

    /**
     * @brief Retrieves the mode of progress for the analysis.
     *
     * @return A map containing the mode of progress for each component.
     */
    map[string, string] progress_mode() except+ nogil

    /**
     * @brief Retrieves the progress report for the analysis.
     *
     * @return A map containing the progress report for each component.
     */
    map[string, string] progress_report() except+ nogil

    /**
     * @brief Checks if the analysis is complete.
     *
     * @return A map indicating the completion status of each component.
     */
    map[string, bool] is_complete() except+ nogil

    /**
     * @brief Retrieves metadata associated with the analysis.
     */
    map[string, meta*] meta_data

    /**
     * @brief Settings for configuring the analysis.
     */
    settings_t m_settings


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

cdef class Analysis:
    cdef list selections_
    cdef list graphs_
    cdef list events_
    cdef list models_
    cdef list optim_
    cdef dict meta_
    cdef analysis* ana


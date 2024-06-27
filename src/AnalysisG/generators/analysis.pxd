# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.event_template cimport event_template
from AnalysisG.core.graph_template cimport graph_template
from AnalysisG.core.selection_template cimport selection_template
from AnalysisG.core.model_template cimport model_template
from AnalysisG.core.lossfx cimport optimizer_params_t
from AnalysisG.core.structs cimport settings_t

cdef extern from "<generators/analysis.h>":

    cdef cppclass analysis:
        analysis() except+

        void add_samples(string path, string label) except +
        void add_selection_template(selection_template* ev) except +
        void add_event_template(event_template* ev, string label) except +
        void add_graph_template(graph_template* ev, string label) except +
        void add_model(model_template* model, optimizer_params_t* op, string run_name) except +
        void start() except +
        void attach_threads() except +

        map[string, float] progress() except +
        map[string, string] progress_mode() except +
        map[string, string] progress_report() except +
        map[string, bool] is_complete() except+

        # settings
        settings_t m_settings

cdef class Analysis:
    cdef analysis* ana

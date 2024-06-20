# distutils: language=c++
# cython: language_level=3

from libcpp.string cimport string
from AnalysisG.core.event_template cimport event_template
from AnalysisG.core.graph_template cimport graph_template
from AnalysisG.core.model_template cimport model_template
from AnalysisG.core.lossfx cimport optimizer_params_t

cdef extern from "<generators/analysis.h>":

    cdef cppclass analysis:
        analysis() except+

        void add_samples(string path, string label) except +
        void add_event_template(event_template* ev, string label) except +
        void add_graph_template(graph_template* ev, string label) except +
        void add_model(model_template* model, optimizer_params_t* op, string run_name) except +
        void start() except +

        # settings
        string output_path

        int epochs
        int kfolds
        int num_examples
        float train_size

cdef class Analysis:
    cdef analysis* ana

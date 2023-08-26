from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool

from cyevent cimport CyEventTemplate, CyGraphTemplate, CySelectionTemplate
from cytypes cimport meta_t, event_t, graph_t, selection_t
from cytypes cimport settings_t, tracer_t, code_t
from cycode cimport CyCode

cdef extern from "../root/root.h" namespace "SampleTracer":
    cdef cppclass CyBatch:
        CyBatch() except +
        string Hash() except +
        void Contextualize() except +

        map[string, CyEventTemplate*] events
        map[string, CyGraphTemplate*] graphs
        map[string, CySelectionTemplate*] selections

        const meta_t* meta

        bool get_event
        bool get_graph
        bool get_selection
        bool valid

        CyEventTemplate* this_ev
        CyGraphTemplate* this_gr
        CySelectionTemplate* this_sel

        string this_tree
        string this_event_name





cdef extern from "../sampletracer/sampletracer.h" namespace "SampleTracer":
    cdef cppclass CySampleTracer:
        CySampleTracer() except +
        void AddEvent(event_t event, meta_t meta, vector[code_t] code) except +

        tracer_t Export() except +
        void Import(tracer_t impt) except +

        vector[CyBatch*] MakeIterable() except +
        map[string, int] length() except +

        settings_t settings
        map[string, CyCode*] code_hashes

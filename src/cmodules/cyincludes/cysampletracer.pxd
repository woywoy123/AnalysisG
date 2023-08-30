from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool

from cyevent cimport CyEventTemplate
from cygraph cimport CyGraphTemplate
from cyselection cimport CySelectionTemplate

from cytypes cimport meta_t, event_t, graph_t, selection_t
from cytypes cimport settings_t, tracer_t, batch_t, code_t
from cycode cimport CyCode

cdef extern from "../root/root.h" namespace "SampleTracer":
    cdef cppclass CyBatch:
        CyBatch() except +
        string Hash() except +
        batch_t Export() except +
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
        void AddEvent(event_t, meta_t) except +
        void AddGraph(graph_t, meta_t) except +
        void AddCode(code_t code) except +

        tracer_t Export() except +
        void Import(tracer_t) except +

        settings_t ExportSettings() except +
        void ImportSettings(settings_t) except +


        vector[CyBatch*] MakeIterable() except +
        map[string, int] length() except +

        CySampleTracer* operator+(CySampleTracer*) except +
        void iadd(CySampleTracer*) except +

        settings_t settings
        map[string, CyCode*] code_hashes
        map[string, string] link_event_code
        map[string, int] event_trees

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

cdef extern from "../root/root.h" namespace "SampleTracer" nogil:
    cdef cppclass CyBatch nogil:
        CyBatch(string) except +
        string Hash() except +

        batch_t ExportPickled() except +
        void ImportPickled(const batch_t*) except +

        batch_t Export() except +
        void Import(const meta_t*) except +
        void Import(const event_t*) except +
        void Import(const graph_t*) except +
        void Import(const selection_t*) except +
        void Import(const batch_t*) except +

        void Contextualize() except +
        void ApplyCodeHash(const map[string, CyCode*]*) except +

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
        map[string, CyCode*] code_hashes

        string this_tree
        string this_event_name
        string hash
        bool lock_meta



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
        map[string, int] event_trees

        map[string, string] link_event_code
        map[string, string] link_graph_code
        string caller

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map

from cyevent cimport CyEventTemplate, CyGraphTemplate, CySelectionTemplate
from cycode cimport CyCode
from cytypes cimport *

cdef extern from "../root/root.h" namespace "SampleTracer":
    cdef cppclass CyBatch:
        CyBatch() except +
        string Hash() except +

        map[string, CyEventTemplate*] events
        map[string, CyGraphTemplate*] graphs
        map[string, CySelectionTemplate*] selections

        const meta_t* meta


        bool get_event
        bool get_graph
        bool get_selection

        bool m_ev
        CyEventTemplate* this_ev

        bool m_gr
        CyGraphTemplate* this_gr

        bool m_sel
        CySelectionTemplate* this_sel

        string this_tree





cdef extern from "../sampletracer/sampletracer.h" namespace "SampleTracer":
    cdef cppclass CySampleTracer:
        CySampleTracer() except +
        void AddEvent(event_t event, meta_t meta, vector[code_t] code) except +

        tracer_t Export() except +
        void Import(tracer_t impt) except +

        vector[CyBatch*] MakeIterable() except +
        map[string, int] length() except +

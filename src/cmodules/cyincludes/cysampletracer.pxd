from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map

from cyevent cimport CyEventTemplate
from cycode cimport CyCode
from cytypes cimport *

cdef extern from "../sampletracer/sampletracer.h" namespace "SampleTracer":
    cdef cppclass CySampleTracer:
        CySampleTracer() except +
        void AddEvent(event_t event, meta_t meta, vector[code_t] code) except +
        void iadd(CySampleTracer* smpl) except +
        void Import(tracer_t impt) except +
        tracer_t Export() except +

from cyevent cimport CyEventTemplate, ExportEventTemplate
from cymetadata cimport ExportMetaData
from cycode cimport CyCode, ExportCode
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map

cdef extern from "../sampletracer/sampletracer.h":
    struct Container:
        vector[string] event_name
        vector[ExportEventTemplate] events
        ExportMetaData meta

cdef extern from "../sampletracer/root.h" namespace "SampleTracer":
    cdef cppclass CyROOT:
        CyROOT(ExportMetaData) except +

cdef extern from "../sampletracer/sampletracer.h" namespace "SampleTracer":
    cdef cppclass CySampleTracer:
        CySampleTracer() except +
        void AddEvent(ExportEventTemplate event, ExportMetaData meta, ExportCode code) except +
        map[string, Container] Search(vector[string] tofind) except +
        map[string, unsigned int] Length() except +

        map[string, CyROOT*] ROOT_map
        map[string, CyCode*] event_code


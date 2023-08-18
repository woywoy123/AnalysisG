
cdef extern from "sampletracer.h" namespace "SampleTracer":
    cdef cppclass CySampleTracer:
        CySampleTracer() except +


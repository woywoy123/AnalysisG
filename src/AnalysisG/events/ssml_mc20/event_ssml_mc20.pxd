# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector

from AnalysisG.events.ssml_mc20.event_ssml_mc20 cimport *
from AnalysisG.core.particle_template cimport particle_template
from AnalysisG.core.event_template cimport *

cdef extern from "<ssm_mc20/event_ssml_mc20.h>":

    cdef cppclass event_ssml_mc20(event_template):
        event_ssml_mc20() except+

cdef class SSML_MC20(EventTemplate):
    pass

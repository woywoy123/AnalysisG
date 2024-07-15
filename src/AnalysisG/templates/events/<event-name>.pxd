# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector

from AnalysisG.events.<particle-module>.<particle-header> cimport *
from AnalysisG.core.particle_template cimport particle_template
from AnalysisG.core.event_template cimport *

cdef extern from "<<event-module>/<event-name>.h>":

    cdef cppclass <event-name>(event_template):
        <event-name>() except+

        vector[particle_template*] some_particles

cdef class <Python-Event>(EventTemplate):
    pass

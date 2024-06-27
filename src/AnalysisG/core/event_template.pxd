# distutils: language=c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.functional cimport function
from cython.operator cimport dereference as deref

from AnalysisG.core.io cimport data_t

cdef extern from "<templates/event_template.h>":
    cdef cppclass event_template:
        event_template() except +

        vector[string] trees
        vector[string] branches
        vector[string] leaves

        string tree
        string name
        string hash

        double weight
        double index

        map[string, event_template*] build_event(map[string, data_t*]* evnt) except +
        void CompileEvent() except+

        bool operator == (event_template& p)

cdef class EventTemplate:
    cdef event_template* ptr


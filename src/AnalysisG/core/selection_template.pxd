# distutils: language=c++
# cython: language_level = 3

from libcpp cimport bool
from libcpp.string cimport string
from AnalysisG.core.event_template cimport event_template

cdef extern from "<templates/selection_template.h>":
    cdef cppclass selection_template:
        selection_template() except +

        string name
        string hash
        string tree
        double index

        selection_template* build(event_template*)
        bool operator == (selection_template& p)

cdef class SelectionTemplate:
    cdef selection_template* ptr
    cdef void transform_dict_keys(self)


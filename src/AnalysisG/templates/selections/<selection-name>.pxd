# distuils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.selection_template cimport *

cdef extern from "<selection-name>.h":
    cdef cppclass <selection-name>(selection_template):
        <selection-name>() except +



cdef class <selection-name-python>(SelectionTemplate):
    cdef <selection-name>* tt




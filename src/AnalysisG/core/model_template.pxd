# distutils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "<templates/model_template.h>" nogil:
    cdef cppclass model_template:

        model_template() except+ nogil

        map[string, string] o_graph
        map[string, string] o_node
        map[string, string] o_edge

        vector[string] i_graph
        vector[string] i_node
        vector[string] i_edge

        string device
        string name
        string model_checkpoint_path

cdef class ModelTemplate:

    cdef model_template* nn_ptr;
    cdef dict conv(self, map[string, string]*)
    cdef map[string, string] cond(self, dict inpt)

# distutils: language=c++
# cython: language_level=3

from libcpp.string cimport string
from libcpp.map cimport map, pair
from AnalysisG.core.tools cimport *
from AnalysisG.core.model_template cimport model_template
from cython.operator cimport dereference as deref

cdef class ModelTemplate:

    def __cinit__(self):
        if type(self) is not ModelTemplate: return
        self.nn_ptr = new model_template()

    def __init__(self): pass

    def __dealloc__(self):
        if type(self) is not ModelTemplate: return
        del self.nn_ptr

    cdef dict conv(self, map[string, string]* inpt):
        cdef dict out = {}
        cdef pair[string, string] itx
        for itx in deref(inpt): out[env(itx.first)] = env(itx.second)
        return out

    cdef map[string, string] cond(self, dict inpt):
        cdef str i
        self.nn_ptr.name = enc(self.__class__.__name__)
        return {enc(i) : enc(inpt[i]) for i in inpt}

    @property
    def o_graph(self): return self.conv(&self.nn_ptr.o_graph)

    @o_graph.setter
    def o_graph(self, dict inpt): self.nn_ptr.o_graph = self.cond(inpt)

    @property
    def o_node(self): return self.conv(&self.nn_ptr.o_node)

    @o_node.setter
    def o_node(self, dict inpt): self.nn_ptr.o_node = self.cond(inpt)

    @property
    def o_edge(self): return self.conv(&self.nn_ptr.o_edge)

    @o_edge.setter
    def o_edge(self, dict inpt): self.nn_ptr.o_edge = self.cond(inpt)

    @property
    def i_graph(self): return env_vec(&self.nn_ptr.i_graph)

    @i_graph.setter
    def i_graph(self, list inpt): self.nn_ptr.i_graph = enc_list(inpt)

    @property
    def i_node(self): return env_vec(&self.nn_ptr.i_node)

    @i_node.setter
    def i_node(self, list inpt): self.nn_ptr.i_node = enc_list(inpt)

    @property
    def i_edge(self): return env_vec(&self.nn_ptr.i_edge)

    @i_edge.setter
    def i_edge(self, list inpt): self.nn_ptr.i_edge = enc_list(inpt)


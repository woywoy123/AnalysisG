# distutils: language=c++
# cython: language_level=3

from libcpp.map cimport map
from libcpp.vector cimport vector
from libcpp.string cimport string
from AnalysisG.core.tools cimport *
from AnalysisG.generators.optimizer cimport optimizer
from AnalysisG.core.model_template cimport ModelTemplate
from AnalysisG.generators.graphgenerator cimport GraphGenerator
from AnalysisG.generators.eventgenerator cimport EventGenerator

cdef class Optimizer:
    def __cinit__(self): self.ev_ptr = new optimizer();
    def __init__(self): pass
    def __dealloc__(self): del self.ev_ptr

    def AddGeneratorPairs(self, EventGenerator ev_gen, GraphGenerator gr_gen):
        ev_gen.CompileEvents()
        gr_gen.AddEvents(ev_gen)
        gr_gen.CompileEvents()
        ev_gen.flush()
        del ev_gen

        self.ev_ptr.create_data_loader(gr_gen.ev_ptr.delegate_data())
        del gr_gen

    def DefineOptimizer(self, str name):
        self.ev_ptr.define_optimizer(enc(name))

    def DefineModel(self, ModelTemplate model):
        self.ev_ptr.define_model(model.nn_ptr)

    def Start(self): self.ev_ptr.start()

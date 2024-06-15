# distutils: language=c++
# cython: language_level=3

from tqdm import tqdm
from libcpp cimport bool
from libcpp.map cimport map, pair
from libcpp.string cimport string
from libcpp.vector cimport vector

from AnalysisG.core.tools cimport *
from AnalysisG.core.graph_template cimport graph_template, GraphTemplate

from AnalysisG.generators.graphgenerator cimport graphgenerator, target_search
from AnalysisG.generators.eventgenerator cimport eventgenerator, EventGenerator

cdef class GraphGenerator:
    def __cinit__(self): self.ev_ptr = new graphgenerator()
    def __init__(self): pass
    def __dealloc__(self): del self.ev_ptr
    def ImportGraph(self, GraphTemplate evn):
        self.graph_types[evn.ptr.name] = evn.ptr

    def CompileEvents(self):
        cdef int i, k
        cdef target_search tg
        cdef vector[event_template*] events_
        cdef map[string, graph_template*] graphs_

        for i in range(self.targets.size()):
            tg = self.targets[i]
            if not self.graph_types.count(tg.graph_name): continue

            events_ = self.event_gen[i].get_event(tg.event_name, tg.tree)
            for k in tqdm(range(events_.size())):
               graphs_[tg.tree] = self.graph_types[tg.graph_name].build_event(events_[k])
               self.ev_ptr.add_graph_template(&graphs_)
        self.ev_ptr.compile()

    def AddEvents(self, EventGenerator evg, str event_name = "", str tree = "", str graph_name = ""):
        cdef target_search tg
        if len(graph_name):
            tg.event_name = enc(event_name)
            tg.tree = enc(tree)
            tg.graph_name = enc(graph_name)
            self.targets.push_back(tg)
            self.event_gen.push_back(evg.ev_ptr)
            return

        cdef pair[string, graph_template*] itr
        for itr in self.graph_types: self.AddEvents(evg, event_name, tree, env(itr.first))



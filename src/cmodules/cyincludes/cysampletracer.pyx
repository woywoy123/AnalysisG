# distuils: language = c++
# cython: language_level = 3

from cysampletracer cimport CySampleTracer, CyBatch

from cyevent cimport CyEventTemplate
from cygraph cimport CyGraphTemplate
from cyselection cimport CySelectionTemplate

from cytypes cimport event_t, graph_t, selection_t, code_t
from cytypes cimport tracer_t, batch_t, meta_t, settings_t

from AnalysisG.Tools import Code
from cycode cimport CyCode

from AnalysisG._cmodules.SelectionTemplate import SelectionTemplate
from AnalysisG._cmodules.EventTemplate import EventTemplate
from AnalysisG._cmodules.GraphTemplate import GraphTemplate
from AnalysisG._cmodules.MetaData import MetaData

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map, pair
from libcpp cimport bool

from typing import Union
import torch
import pickle
import os

cdef string enc(str val): return val.encode("UTF-8")
cdef str env(string val): return val.decode("UTF-8")

cdef class Event:

    cdef CyBatch* ptr
    cdef meta_t m_meta

    cdef _event
    cdef _graph
    cdef _selection
    cdef _meta
    cdef _owner

    def __cinit__(self):
        self.ptr = NULL
        self._event = None
        self._selection = None
        self._graph = None
        self._meta = None
        self._owner = False

    def __init__(self): pass

    def __eq__(self, other):
        try: return self.hash == other.hash
        except: return False

    def __hash__(self):
        return int(self.hash[:8], 0)

    def __dealloc__(self):
        if not self._owner: return
        else: del self.ptr

    def __getattr__(self, req):
        self.__getevent__()
        try: return getattr(self._event, req)
        except AttributeError: pass

        self.__getgraph__()
        try: return getattr(self._graph, req)
        except AttributeError: pass

        self.__getselection__()
        try: return getattr(self._selection, req)
        except AttributeError: pass

        self.__getmeta__()
        try: return getattr(self._meta, req)
        except AttributeError: pass

    def meta(self):
        return self._meta

    def __getmeta__(self):
        if self._meta is not None: return
        if not self.ptr.lock_meta: return
        self._meta = MetaData()
        self._meta.__setstate__(self.ptr.meta[0])

    def __getevent__(self):
        if self._event is not None: return
        cdef CyEventTemplate* ev_ = self.ptr.this_ev
        if ev_ == NULL: self._event = False; return
        cdef CyCode* co_ = ev_.code_link
        if co_ == NULL: print("EVENT -> MISSING CODE!"); return

        cdef event_t ev = ev_.Export()
        cdef code_t  co = co_.ExportCode()
        cdef int c_i, d_i
        c_i = co_.dependency.size()
        d_i = co_.container.dependency_hashes.size()
        if c_i < d_i: print("MISSING DEPENDENCY CODE!"); return

        c = Code()
        c.__setstate__(co)
        cdef pair[string, CyCode*] itr
        for itr in co_.dependency:
            co = itr.second.ExportCode()
            c.AddDependency([co])
        event = c.InstantiateObject

        cdef dict pkl = pickle.loads(ev.pickled_data)
        ev.pickled_data = b""
        event.__setstate__((pkl, ev))
        self._event = event

    def __getgraph__(self):
        if self._graph is not None: return
        cdef CyGraphTemplate* gr_ = self.ptr.this_gr
        if gr_ == NULL: self._graph = False; return
        cdef CyCode* co_ = gr_.code_link
        if co_ == NULL: print("GRAPH -> MISSING CODE!"); return
        c = Code()
        c.__setstate__(co_.ExportCode())
        gr = c.InstantiateObject
        if self._event is None: self._graph = gr()
        elif not self._event: self._graph = gr()
        else: self._graph = gr(self._event)
        self._graph.Import(gr_.Export())


    def __getselection__(self):
        if self._selection is not None: return
        cdef CySelectionTemplate* sel_ = self.ptr.this_sel
        if sel_ == NULL: self._selection = False; return
        cdef CyCode* co_ = sel_.code_link
        if co_ == NULL: print("SELECTION -> MISSING CODE!"); return
        c = Code()
        c.__setstate__(co_.ExportCode())
        sel = c.InstantiateObject
        sel.__setstate__(sel_.Export())
        self._selection = sel

    def __getstate__(self) -> tuple:
        return (self.ptr.meta[0], self.ptr.ExportPickled())

    def __setstate__(self, tuple inpt):
        cdef batch_t b = inpt[1]
        self.m_meta = inpt[0]
        self.ptr = new CyBatch(b.hash)
        self.ptr.Import(&self.m_meta)
        self.ptr.ImportPickled(&b)
        self._owner = True


cdef class SampleTracer:

    cdef CySampleTracer* ptr
    cdef _Event
    cdef _Graph
    cdef dict _Selections
    cdef int b_end
    cdef int b_start
    cdef int _nhashes
    cdef settings_t* _set
    cdef vector[CyBatch*] batches

    def __cinit__(self):
        self.ptr = new CySampleTracer()
        self._set = &self.ptr.settings
        self._Event = None
        self._Graph = None
        self._Selections = {}
        self._nhashes = 0
        self.b_start = 0
        self.b_end = 0

    def __dealloc__(self):
        del self.ptr

    def __init__(self):
        pass

    def __getstate__(self) -> tracer_t:
        return self.ptr.Export()

    def __setstate__(self, tracer_t inpt):
        self.ptr.Import(inpt)

    def __getitem__(self, key: Union[list, str]):
        cdef vector[string] inpt;
        cdef str it
        if isinstance(key, str): inpt = [enc(key)]
        else: inpt = [enc(it) for it in key]
        self._set.search = inpt
        cdef list output = self.makelist()
        self._set.search.clear()

        if not len(output): return False
        return output[0] if len(output) == 1 else output

    def __contains__(self, str val) -> bool:
        return self.__getitem__(val) != False

    def __len__(self) -> int:
        cdef map[string, int] f = self.ptr.length()
        cdef pair[string, int] it
        cdef int entries = 0
        cdef str name = ""
        if len(self.Tree): name += self.Tree
        if len(self.EventName): name += "/" + self.EventName
        if not len(name):
            dc = {env(it.first) : it.second for it in f}
            self._nhashes = dc["n_hashes"]
            del dc["n_hashes"]
            return sum([entries for entries in dc.values()])

        for it in f:
            if name not in env(it.first): continue
            entries += it.second
        return entries

    def __add__(self, SampleTracer other) -> SampleTracer:
        cdef SampleTracer tr = self.clone()
        tr.ptr.iadd(self.ptr)
        tr.ptr.iadd(other.ptr)
        return tr

    def __radd__(self, other) -> SampleTracer:
        if self.is_self(other): return self.__add__(other)
        return self.__add__(self.clone())

    def __iadd__(self, other) -> SampleTracer:
        if not self.is_self(other): return self
        cdef SampleTracer tr = other
        self.ptr.iadd(tr.ptr)
        return self

    def __iter__(self):
        if self.preiteration(): return self
        self.batches = self.ptr.MakeIterable()
        self.b_end = self.batches.size()
        self.b_start = 0
        return self

    def __next__(self) -> Event:
        if self.b_end == self.b_start: raise StopIteration
        cdef Event event = Event()
        event.ptr = self.batches[self.b_start]
        event.ptr.Import(self.batches[self.b_start].meta)
        self.b_start += 1
        return event


    # ------------------ CUSTOM FUNCTIONS ------------------ #
    def trace_code(self, obj) -> code_t:
        cdef code_t co = Code(obj).__getstate__()
        self.ptr.AddCode(co)
        return co

    def ImportSettings(self, settings_t inpt):
        self.ptr.ImportSettings(inpt)

    def ExportSettings(self) -> settings_t:
        return self.ptr.ExportSettings()

    def clone(self):
        return self.__class__()

    def is_self(self, inpt, obj = SampleTracer) -> bool:
        return issubclass(inpt.__class__, obj)

    def preiteration(self) -> bool:
        return False

    def makelist(self) -> list:
        cdef vector[CyBatch*] evnt = self.ptr.MakeIterable()
        cdef CyBatch* batch
        cdef Event event
        cdef list output = []
        for batch in evnt:
            event = Event()
            event.__setstate__((batch.meta[0], batch.ExportPickled()))
            output.append(event)
        return output

    def AddEvent(self, event_inpt, meta_inpt = None):
        cdef event_t event
        cdef code_t co
        cdef string name
        cdef tuple get

        if meta_inpt is not None:
            get = event_inpt.__getstate__()
            event = get[1]
            event.pickled_data = pickle.dumps(get[0])
            name = event.event_name

            self.ptr.event_trees[event.event_tree] += 1
            if not self.ptr.link_event_code.count(name): self.Event = event_inpt
            self.ptr.AddEvent(event, meta_inpt.__getstate__())
            return

        cdef str g
        cdef dict ef
        cdef list evnts = [ef for g in event_inpt for ef in event_inpt[g].values()]
        for ef in evnts: self.AddEvent(ef["Event"], ef["MetaData"])


    def AddGraph(self, graph_inpt, meta_inpt = None):
        if graph_inpt is None: return
        if meta_inpt is None: self.ptr.AddGraph(graph_inpt.__getstate__(), meta_t())
        else: self.ptr.AddGraph(graph_inpt.__getstate__(), meta_inpt.__getstate__())

    def AddSelections(self, selection_inpt, meta_inpt = None):
        if selection_inpt is None: return
        if meta_inpt is None: self.ptr.AddSelection(selection_inpt.__getstate__(), meta_t())
        else: self.ptr.AddSelection(selection_inpt.__getstate__(), meta_inpt.__getstate__())

    def SetAttribute(self, fx, str name) -> bool:
        if self._Graph is None: self._Graph = GraphTemplate()
        if name in self._Graph.code: return False
        self._Graph.__scrapecode__(fx, name)
        return True

    @property
    def Event(self):
        return self._Event

    @Event.setter
    def Event(self, event):
        try: event = event()
        except: pass
        if not self.is_self(event, EventTemplate): return
        cdef string name = enc(event.__name__())
        cdef code_t co
        cdef map[string, code_t] deps = {}
        for o in event.Objects.values():
            co = self.trace_code(o)
            deps[co.hash] = co
        co = self.trace_code(event)
        self.ptr.link_event_code[name] = co.hash
        self.ptr.code_hashes[co.hash].AddDependency(deps)
        self._Event = event

    @property
    def Graph(self):
        return self._Graph

    @Graph.setter
    def Graph(self, graph):
        try: graph = graph()
        except: pass
        if not self.is_self(graph, GraphTemplate): return
        cdef graph_t gr
        cdef string name = enc(graph.__name__())
        if self._Graph is not None:
            gr = self._Graph.Export
            gr.event_name = name
            graph.Import(gr)
            graph.ImportCode(self._Graph.code)
        for _, o in graph.code.items(): self.ptr.AddCode(o.__getstate__())
        cdef code_t co = self.trace_code(graph)
        self.ptr.link_graph_code[name] = co.hash
        self._Graph = graph

    @property
    def Selections(self):
        return self._Selections

    @Selections.setter
    def Selections(self, selection):
        try: selection = selection()
        except: pass
        if not self.is_self(selection, SelectionTemplate): return
        cdef code_t co = self.trace_code(selection)
        self.ptr.link_selection_code[co.class_name] = co.hash
        self._Selections[env(co.class_name)] = selection

    @property
    def ShowEvents(self) -> list:
        cdef list out = []
        cdef pair[string, string] it
        cdef map[string, string] ev = self.ptr.link_event_code
        for it in ev: out.append(env(it.first))
        return out

    @property
    def ShowGraphs(self) -> list:
        cdef list out = []
        cdef pair[string, string] it
        cdef map[string, string] ev = self.ptr.link_graph_code
        for it in ev: out.append(env(it.first))
        return out

    @property
    def ShowSelections(self) -> list:
        cdef list out = []
        cdef pair[string, string] it
        cdef map[string, string] ev = self.ptr.link_selection_code
        for it in ev: out.append(env(it.first))
        return out

    @property
    def ShowLength(self) -> dict:
        cdef pair[string, int] it
        cdef dict output = {}
        for it in self.ptr.length():
            if env(it.first) == "n_hashes": self._nhashes = it.second
            else: output[env(it.first)] = it.second
        return output

    @property
    def ShowTrees(self) -> list:
        cdef pair[string, int] it
        cdef map[string, int] ev = self.ptr.event_trees
        cdef list out = []
        for it in ev: out.append(env(it.first))
        return out

    @property
    def Files(self) -> dict:
        cdef dict output = {}
        cdef string k
        cdef pair[string, vector[string]] it
        for it in self._set.files: output[env(it.first)] = [env(k) for k in  it.second]
        return output

    @Files.setter
    def Files(self, val: Union[str, list, dict]):
        cdef dict Files = {}
        cdef str key, k
        if isinstance(val, str): Files["None"] = [val]
        elif isinstance(val, list): Files["None"] = val
        else: Files.update(val)
        if not len(Files): self._set.files.clear();
        for key in Files:
            for k in Files[key]: self._set.files[enc(key)].push_back(enc(k))

    @property
    def Threads(self) -> int:
        return self._set.threads

    @Threads.setter
    def Threads(self, int val):
        self._set.threads = val

    @property
    def GetEvent(self) -> bool:
        return self._set.getevent

    @GetEvent.setter
    def GetEvent(self, bool val):
        self._set.getevent = val

    @property
    def GetGraph(self) -> bool:
        return self._set.getgraph

    @GetGraph.setter
    def GetGraph(self, bool val):
        self._set.getgraph = val

    @property
    def GetSelection(self) -> bool:
        return self._set.getselection

    @GetSelection.setter
    def GetSelection(self, bool val):
        self._set.getselection = val

    @property
    def ProjectName(self) -> str:
        return env(self._set.projectname)

    @ProjectName.setter
    def ProjectName(self, str val):
        self._set.projectname = enc(val)

    @property
    def EventName(self) -> str:
        return env(self._set.eventname)

    @EventName.setter
    def EventName(self, val: Union[str, None]):
        if val is None: val = "NULL"
        self._set.eventname = enc(val)

    @property
    def GraphName(self) -> str:
        return env(self._set.graphname)

    @GraphName.setter
    def GraphName(self, val: Union[str, None]):
        if val is None: val = "NULL"
        self._set.graphname = enc(val)

    @property
    def SelectionName(self) -> str:
        return env(self._set.selectionname)

    @SelectionName.setter
    def SelectionName(self, val: Union[str, None]):
        if val is None: val = "NULL"
        self._set.selectionname = enc(val)

    @property
    def Tree(self) -> str:
        return env(self._set.tree)

    @Tree.setter
    def Tree(self, str val):
        self._set.tree = enc(val)

    @property
    def ProjectName(self) -> str:
        return env(self._set.projectname)

    @ProjectName.setter
    def ProjectName(self, str val):
        self._set.projectname = enc(val)

    @property
    def OutputDirectory(self) -> str:
        return env(self._set.outputdirectory)

    @OutputDirectory.setter
    def OutputDirectory(self, str val):
        if not val.endswith("/"): val += "/"
        val = os.path.abspath(val)
        self._set.outputdirectory = enc(val)

    @property
    def Caller(self) -> str:
        return env(self.ptr.caller)

    @Caller.setter
    def Caller(self, str val):
        self.ptr.caller = enc(val.upper())

    @property
    def EventStart(self):
        return self._set.event_start

    @EventStart.setter
    def EventStart(self, x: Union[int, None]):
        if x is None: x = 0
        self._set.event_start = x

    @property
    def EventStop(self):
        if self._set.event_stop == 0: return None
        return self._set.event_stop

    @EventStop.setter
    def EventStop(self, x: Union[int, None]):
        if x is None: x = 0
        self._set.event_stop = x

    @property
    def Verbose(self):
        return self._set.verbose

    @Verbose.setter
    def Verbose(self, int val):
        self._set.verbose = val

    @property
    def Chunks(self):
        return self._set.chunks

    @Chunks.setter
    def Chunks(self, int val):
        self._set.chunks = val

    @property
    def EnablePyAMI(self):
        return self._set.enable_pyami

    @EnablePyAMI.setter
    def EnablePyAMI(self, bool val):
        self._set.enable_pyami = val

    @property
    def Device(self):
        return env(self._set.device)

    @Device.setter
    def Device(self, str val):
        self._set.device = enc(val)

    @property
    def nHashes(self) -> int:
        return self._nhashes

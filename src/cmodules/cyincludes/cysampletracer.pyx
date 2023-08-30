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

from AnalysisG._cmodules.EventTemplate import EventTemplate
from AnalysisG._cmodules.GraphTemplate import GraphTemplate
from AnalysisG._cmodules.MetaData import MetaData

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map, pair
from libcpp cimport bool

from typing import Union
import pickle
import os

cdef string enc(str val): return val.encode("UTF-8")
cdef str env(string val): return val.decode("UTF-8")

cdef class Event:

    cdef CyBatch* ptr
    cdef _event
    cdef _graph
    cdef _selection
    cdef _meta

    def __cinit__(self):
        self.ptr = NULL
        self._event = None
        self._selection = None
        self._graph = None
        self._meta = None

    def __init__(self): pass

    def __eq__(self, other):
        try: return self.hash == other.hash
        except: return False

    def __hash__(self):
        return int(self.hash[:8], 0)

    def __getattr__(self, req):
        self.__getevent__()
        try: return getattr(self._event, req)
        except AttributeError: pass

        self.__getgraph__()
        try: return getattr(self._graph, req)
        except AttributeError: pass

        self.__getmeta__()
        try: return getattr(self._meta, req)
        except AttributeError: pass

    def __getmeta__(self):
        if self._meta is not None: return
        cdef meta_t meta = self.ptr.meta[0]
        self._meta = MetaData()
        self._meta.__setstate__(meta)

    def __getevent__(self):
        if self._event is not None: return
        cdef CyEventTemplate* ev_ = self.ptr.this_ev
        if ev_ == NULL: self._event = False; return
        cdef CyCode* co_ = ev_.code_link
        if co_ == NULL: print("MISSING CODE!"); return

        cdef event_t ev = ev_.Export()
        cdef code_t  co = co_.ExportCode()
        cdef pair[string, CyCode*] itr
        cdef list deps = []
        for itr in co_.dependency:
            deps.append(itr.second.ExportCode())
        c = Code()
        c.__setstate__(co)
        c.AddDependency(deps)
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
        if co_ == NULL: print("MISSING CODE!"); return
        c = Code()
        c.__setstate__(co_.ExportCode())
        graph = c.InstantiateObject
        graph.Import(gr_.Export())
        self._graph = graph

    def __getstate__(self) -> tuple:
        cdef meta_t meta = self.ptr.meta[0]
        return (meta, self.ptr.Export())


cdef class SampleTracer:

    cdef CySampleTracer* ptr
    cdef event_iter
    cdef _Event
    cdef _Graph

    def __cinit__(self):
        self.ptr = new CySampleTracer()
        self._Event = None
        self._Graph = None

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
        self.ptr.settings.search = inpt
        cdef list output = self.makelist()
        self.ptr.settings.search.clear()

        if not len(output): return False
        return output[0] if len(output) == 1 else output

    def __contains__(self, str val) -> bool:
        return self.__getitem__(val) != False

    def __len__(self) -> int:
        cdef map[string, int] f = self.ptr.length()
        cdef pair[string, int] it

        cdef str name = ""
        if len(self.Tree): name += self.Tree
        if len(self.EventName): name += "/" + self.EventName
        if not len(name): return sum([it.second for it in f])

        cdef int entries = 0
        for it in f:
            if name not in env(it.first): continue
            entries += it.second
        return entries

    def __add__(self, SampleTracer other) -> SampleTracer:
        cdef SampleTracer tr = SampleTracer()
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
        self.event_iter = iter(self.makelist())
        return self

    def __next__(self) -> Event:
        return next(self.event_iter)


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
        cdef string name
        cdef pair[string, int] itr
        cdef pair[string, string] it
        cdef map[string, int] tree = self.ptr.event_trees
        cdef map[string, string] ev = self.ptr.link_event_code

        # Case where no event implementation has been specified 
        if not self.ptr.settings.eventname.size():
            for it in ev: name = it.first; break
            self.ptr.settings.eventname = name

        # Case where no event tree has been specified
        if not self.ptr.settings.tree.size():
            for itr in tree: name = itr.first; break
            self.ptr.settings.tree = name
        return False

    def makelist(self) -> list:
        cdef vector[CyBatch*] evnt = self.ptr.MakeIterable()
        cdef CyBatch* batch
        cdef Event event
        cdef list output = []
        for batch in evnt:
            event = Event()
            event.ptr = batch
            output.append(event)
        return output

    def AddEvent(self, event_inpt, meta_inpt = None):
        cdef meta_t meta
        cdef event_t event
        cdef code_t co
        cdef string name
        cdef tuple get

        if meta_inpt is not None:
            meta = meta_inpt.__getstate__()
            get = event_inpt.__getstate__()

            event = get[1]
            event.pickled_data = pickle.dumps(get[0])
            name = event.event_name

            self.ptr.event_trees[event.event_tree] += 1
            if not self.ptr.link_event_code.count(name):
                self.Event = event_inpt
            self.ptr.AddEvent(event, meta)
            return

        cdef str g
        cdef dict ef
        cdef list evnts = [ef for g in event_inpt for ef in event_inpt[g].values()]
        for ef in evnts: self.AddEvent(ef["Event"], ef["MetaData"])


    def AddGraph(self, graph_inpt, meta_t meta):
        cdef graph_t graph = graph_inpt.__getstate__()
        self.ptr.AddGraph(graph, meta)

    def SetAttribute(self, fx, str name, str type_) -> tuple:
        cdef code_t co = self.trace_code(fx)
        cdef string name_fx = co.function_name
        if len(name) > 2: name_fx = env(name[:2] + name)

        if "G" == type_:
            if self.ptr.settings.graph_attribute.count(name_fx): return (name_fx, False)
            self.ptr.settings.graph_attribute[name_fx] = co
        elif "N" == type_:
            if self.ptr.settings.node_attribute.count(name_fx): return (name_fx, False)
            self.ptr.settings.node_attribute[name_fx] = co
        elif "E" == type_:
            if self.ptr.settings.edge_attribute.count(name_fx): return (name_fx, False)
            self.ptr.settings.edge_attribute[name_fx] = co
        return (name_fx, True)


    @property
    def Event(self):
        return self._Event

    @Event.setter
    def Event(self, inpt):
        try: event = inpt()
        except: event = inpt
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
    def Graph(self, inpt):
        try: graph = inpt()
        except: graph = inpt
        if not self.is_self(graph, GraphTemplate): return
        cdef string name = enc(graph.__name__())
        for _, o in graph.code.items(): self.ptr.AddCode(o.__getstate__())
        cdef code_t co = self.trace_code(graph)
        self.ptr.link_event_code[name] = co.hash
        self._Graph = graph

    @property
    def ShowEvents(self) -> list:
        cdef pair[string, string] it
        cdef map[string, string] ev = self.ptr.link_event_code
        cdef list out = []
        for it in ev: out.append(env(it.first))
        return out

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
        for it in self.ptr.settings.files: output[env(it.first)] = [env(k) for k in  it.second]
        return output

    @Files.setter
    def Files(self, val: Union[str, list, dict]):
        cdef dict Files = {}
        cdef str key, k
        if isinstance(val, str): Files["None"] = [val]
        elif isinstance(val, list): Files["None"] = val
        else: Files.update(val)
        if not len(Files): self.ptr.settings.files.clear();
        for key in Files:
            for k in Files[key]: self.ptr.settings.files[enc(key)].push_back(enc(k))

    @property
    def Threads(self) -> int:
        return self.ptr.settings.threads

    @Threads.setter
    def Threads(self, int val):
        self.ptr.settings.threads = val

    @property
    def GetSelection(self) -> bool:
        return self.ptr.settings.getselection

    @GetSelection.setter
    def GetSelection(self, bool val):
        self.ptr.settings.getselection = val

    @property
    def GetEvent(self) -> bool:
        return self.ptr.settings.getevent

    @GetEvent.setter
    def GetEvent(self, bool val):
        self.ptr.settings.getevent = val

    @property
    def GetGraph(self) -> bool:
        return self.ptr.settings.getgraph

    @GetGraph.setter
    def GetGraph(self, bool val):
        self.ptr.settings.getgraph = val

    @property
    def ProjectName(self) -> str:
        return env(self.ptr.settings.projectname)

    @ProjectName.setter
    def ProjectName(self, str val):
        self.ptr.settings.projectname = enc(val)

    @property
    def EventName(self) -> str:
        return env(self.ptr.settings.eventname)

    @EventName.setter
    def EventName(self, str val):
        self.ptr.settings.eventname = enc(val)

    @property
    def Tree(self) -> str:
        return env(self.ptr.settings.tree)

    @Tree.setter
    def Tree(self, str val):
        self.ptr.settings.tree = enc(val)

    @property
    def ProjectName(self) -> str:
        return env(self.ptr.settings.projectname)

    @ProjectName.setter
    def ProjectName(self, str val):
        self.ptr.settings.projectname = enc(val)

    @property
    def OutputDirectory(self) -> str:
        return env(self.ptr.settings.outputdirectory)

    @OutputDirectory.setter
    def OutputDirectory(self, str val):
        if not val.endswith("/"): val += "/"
        val = os.path.abspath(val)
        self.ptr.settings.outputdirectory = enc(val)

    @property
    def Caller(self) -> str:
        return env(self.ptr.settings.caller)

    @Caller.setter
    def Caller(self, str val):
        self.ptr.settings.caller = enc(val.upper())

    @property
    def EventStart(self):
        return self.ptr.settings.event_start

    @EventStart.setter
    def EventStart(self, x: Union[int, None]):
        if x is None: x = 0
        self.ptr.settings.event_start = x

    @property
    def EventStop(self):
        if self.ptr.settings.event_stop == 0: return None
        return self.ptr.settings.event_stop

    @EventStop.setter
    def EventStop(self, x: Union[int, None]):
        if x is None: x = 0
        self.ptr.settings.event_stop = x

    @property
    def Verbose(self):
        return self.ptr.settings.verbose

    @Verbose.setter
    def Verbose(self, int val):
        self.ptr.settings.verbose = val

    @property
    def Chunks(self):
        return self.ptr.settings.chunks

    @Chunks.setter
    def Chunks(self, int val):
        self.ptr.settings.chunks = val

    @property
    def EnablePyAMI(self):
        return self.ptr.settings.enable_pyami

    @EnablePyAMI.setter
    def EnablePyAMI(self, bool val):
        self.ptr.settings.enable_pyami = val


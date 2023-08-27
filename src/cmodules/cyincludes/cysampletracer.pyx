# distuils: language = c++
# cython: language_level = 3

from cysampletracer cimport CySampleTracer, CyBatch

from cyevent cimport CyEventTemplate, CyGraphTemplate, CySelectionTemplate
from cytypes cimport meta_t, settings_t, event_t, graph_t, selection_t, code_t
from cytypes cimport event_T

from AnalysisG.Tools import Code
from cycode cimport CyCode

from AnalysisG._cmodules.MetaData import MetaData

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map, pair
from libcpp cimport bool

from typing import Union

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
    def __getattr__(self, req):
        self.__getevent__()
        try: return getattr(self._event, req)
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
        cdef string code_h = ev_.event.event_code_hash
        cdef CyCode* co_ = ev_.this_code[code_h]

        cdef event_T ev = ev_.Export()
        cdef code_t  co = co_.ExportCode()

        c = Code()
        c.__setstate__(co)
        event = c.InstantiateObject

        event.__setstate__(ev)
        self._event = event


cdef class SampleTracer:

    cdef CySampleTracer* ptr
    cdef map[string, vector[code_t]] hashed_code
    cdef map[string, string] event_code_hash
    cdef map[string, int] event_trees
    cdef vector[CyBatch*] itb
    cdef unsigned int its
    cdef unsigned int ite

    def __cinit__(self):
        self.ptr = new CySampleTracer()
        self.hashed_code = {}
        self.event_code_hash = {}

    def __dealloc__(self): del self.ptr
    def __init__(self): pass

    def __contains__(self, str val) -> bool:
        return self.__getitem__(val) != False

    def __getitem__(self, key: Union[list, str]):
        cdef vector[string] inpt;
        cdef str it
        if isinstance(key, str): inpt = [enc(key)]
        else: inpt = [enc(it) for it in key]
        self.ptr.settings.search = inpt
        cdef list out = [i for i in self]
        self.ptr.settings.search.clear()
        if not len(out): return False
        return out[0] if len(out) == 1 else out

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

    def __add__(self, SampleTracer other):
        cdef SampleTracer tr = SampleTracer()
        tr.ptr.iadd(self.ptr)
        tr.ptr.iadd(other.ptr)

        cdef pair[string, vector[code_t]] itc
        cdef pair[string, string] its
        cdef pair[string, int] iti

        tr.hashed_code = self.hashed_code
        for itc in other.hashed_code:
            tr.hashed_code[itc.first] = itc.second

        tr.event_code_hash = self.event_code_hash
        for its in other.event_code_hash:
            tr.event_code_hash[its.first] = its.second

        tr.event_trees = self.event_trees
        for iti in other.event_trees:
            tr.event_trees[iti.first] += iti.second

        return tr

    def __radd__(self, other):
        if self.is_self(other): return self.__add__(other)
        return self.__add__(self.clone())

    def __iadd__(self, other):
        if not self.is_self(other): return self
        cdef SampleTracer tr = other
        self.ptr.iadd(tr.ptr)

    def __iter__(self):
        if self.__preiteration__(): return self
        self.itb = self.ptr.MakeIterable()
        self.its = 0
        self.ite = self.itb.size()
        return self

    def __next__(self) -> Event:
        if self.its == self.ite: raise StopIteration
        cdef CyBatch* b = self.itb[self.its]
        cdef Event ev = Event()
        ev.ptr = b
        self.its+=1
        return ev


    # Custom internal functions
    def clone(self): return self.__class__()
    def is_self(self, inpt) -> bool:
        return issubclass(inpt.__class__, SampleTracer)

    def __preiteration__(self) -> bool:
        cdef string name
        cdef pair[string, int] itr
        cdef pair[string, string] it
        cdef map[string, int] tree = self.event_trees
        cdef map[string, string] ev = self.event_code_hash

        # Case where no event implementation has been specified 
        if not self.ptr.settings.eventname.size():
            for it in ev: name = it.first; break
            self.ptr.settings.eventname = name


        # Case where no event tree has been specified
        if not self.ptr.settings.tree.size():
            for itr in tree: name = itr.first; break
            self.ptr.settings.tree = name
        return False

    def __scrapecode__(self, event, str event_name):
        cdef string name = enc(event_name)
        if self.hashed_code.count(name): return
        cdef code_t it
        cdef vector[code_t] code = []
        for o in event.Objects.values():
            it = Code(o).__getstate__()
            code.push_back(it)

        it = Code(event).__getstate__()
        code.push_back(it)

        self.hashed_code[name] = code
        self.event_code_hash[name] = it.hash

    def AddEvent(self, event_inpt, meta_inpt = None):
        cdef meta_t meta
        cdef event_T event_
        cdef event_t event
        cdef string name
        cdef vector[code_t] co

        if meta_inpt is not None:
            event_ = event_inpt.__getstate__()
            event = event_.event
            meta = meta_inpt.__getstate__()
            name = event.event_name
            self.__scrapecode__(event_inpt, env(name))

            event.event_code_hash = self.event_code_hash[name]
            self.ptr.AddEvent(event, meta, self.hashed_code[name])
            self.event_trees[event.event_tree] += 1
            return

        cdef str g
        cdef dict ef
        cdef list evnts = [ef for g in event_inpt for ef in event_inpt[g].values()]
        for ef in evnts: self.AddEvent(ef["Event"], ef["MetaData"])

    @property
    def ShowEvents(self) -> list: 
        cdef pair[string, string] it
        cdef map[string, string] ev = self.event_code_hash
        cdef list out = []
        for it in ev: out.append(env(it.first))
        return out
    @property
    def ShowTreeEvents(self) -> list:
        cdef pair[string, int] it
        cdef map[string, int] ev = self.event_tree
        cdef list out = []
        for it in ev: out.append(env(it.first))
        return out

    @property
    def threads(self) -> int: return self.ptr.settings.threads
    @threads.setter
    def threads(self, int val): self.ptr.settings.threads = val

    @property
    def GetSelection(self) -> bool: return self.ptr.settings.getselection
    @GetSelection.setter
    def GetSelection(self, bool val): self.ptr.settings.getselection = val

    @property
    def GetEvent(self) -> bool: return self.ptr.settings.getevent
    @GetEvent.setter
    def GetEvent(self, bool val): self.ptr.settings.getevent = val

    @property
    def GetGraph(self) -> bool: return self.ptr.settings.getgraph
    @GetGraph.setter
    def GetGraph(self, bool val): self.ptr.settings.getgraph = val

    @property
    def ProjectName(self) -> str: return env(self.ptr.settings.projectname)
    @ProjectName.setter
    def ProjectName(self, str val): self.ptr.settings.projectname = enc(val)

    @property
    def EventName(self) -> str: return env(self.ptr.settings.eventname)
    @EventName.setter
    def EventName(self, str val): self.ptr.settings.eventname = enc(val)

    @property
    def Tree(self) -> str: return env(self.ptr.settings.tree)
    @Tree.setter
    def Tree(self, str val): self.ptr.settings.tree = enc(val)








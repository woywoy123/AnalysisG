# distutils: language = c++
# cython: language_level = 3

from cyselection cimport CySelectionTemplate
from cytypes cimport event_t, selection_t
from cytypes cimport code_t

from AnalysisG.Tools import Code

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool

import pickle

cdef string enc(str val): return val.encode("UTF-8")
cdef str env(string val): return val.decode("UTF-8")

cdef class SelectionTemplate:

    cdef CySelectionTemplate* ptr
    cdef selection_t* sel
    cdef _params_

    def __cinit__(self):
        self.ptr = new CySelectionTemplate()
        self.sel = &(self.ptr.selection)
        cdef string name = enc(self.__class__.__name__)
        self.ptr.set_event_name(self.sel, name)
        if not self.sel._params_.size(): self._params_ = None
        else: self._params_ = pickle.loads(self.sel._params_)

    def __init__(self): pass
    def __dealloc__(self): del self.ptr
    def __name__(self) -> str: return env(self.sel.event_name)
    def __hash__(self) -> int: return int(self.hash[:8], 0)

    def __eq__(self, other) -> bool:
        if not self.is_self(other): return False
        cdef SelectionTemplate o = other
        return self.ptr[0] == o.ptr[0]

    def __radd__(self, other):
        if other == 0: return self
        else: self.__add__(other)

    def __add__(self, SelectionTemplate other):
        cdef SelectionTemplate o = self.clone()
        o.ptr.iadd(other.ptr)
        o.ptr.iadd(self.ptr)
        return o

    def __getstate__(self) -> selection_t:
        cdef str key
        cdef dict pkl = {}
        for key in list(self.__dict__):
            pkl[key] = self.__dict__[key]
        self.ptr.selection.pickled_data = pickle.dumps(pkl)
        return self.ptr.Export()

    def __setstate__(self, selection_t inpt):
        self.ptr.Import(inpt)
        if not inpt.pickled_data.size(): return
        cdef str key
        cdef dict pkls = pickle.loads(inpt.pickled_data)
        for key in pkls:
            try: self.__dict__[key] = pkls[key]
            except KeyError: setattr(self, key, pkls[key])

    def __scrapecode__(self):
        co = Code(self)
        cdef code_t code = co.__getstate__()
        self.sel.code_hash = code.hash
        return code

    def is_self(self, inpt) -> bool:
        if isinstance(inpt, SelectionTemplate): return True
        return issubclass(inpt.__class__, SelectionTemplate)

    def clone(self) -> SelectionTemplate:
        return self.__class__()

    def selection(self, event):
        return True

    def Strategy(self, event):
        return True

    def __select__(self, event):
        res = self.selection(event)
        if res is None: res = ""
        if isinstance(res, str): return self.ptr.CheckSelection(enc(str(res)))
        if type(res).__name__ == "bool": return self.ptr.CheckSelection(<bool>res)
        return True

    def __strategy__(self, event):
        self.ptr.StartTime()
        res = self.Strategy(event)
        self.ptr.EndTime()

        if self._params_ is None: pass
        else: self.sel._params_ = pickle.dumps(self._params_)

        if res is None: res = ""
        if isinstance(res, str): return self.ptr.CheckStrategy(enc(str(res)))
        if type(res).__name__ == "bool": return self.ptr.CheckStrategy(<bool>res)
        self.sel.pickled_strategy_data = pickle.dumps(res)
        return True

    def __processing__(self, event):
        cdef event_t ev
        ev.event_index   = event.index
        ev.event_hash    = enc(event.hash)
        ev.event_tagging = enc(event.Tag)
        ev.event_tree    = enc(event.Tree)
        ev.event_root    = enc(event.ROOT)
        ev.weight        = event.weight
        self.ptr.RegisterEvent(&ev)

        if not self.sel.allow_failure:
            if self.__select__(event): pass
            else: return False
            if self.__strategy__(event): return True
            else: return False

        try:
            if not self.__select__(event): return False
        except Exception as inst:
            self.sel.errors[enc(str(inst))] += 1
            return self.ptr.CheckSelection(enc(str(inst)+"::Error"))

        try: return self.__strategy__(event)
        except Exception as inst:
            self.sel.errors[enc(str(inst))] += 1
            return self.ptr.CheckStrategy(enc(str(inst)+"::Error"))


    @property
    def __params__(self): return self._params_

    @__params__.setter
    def __params__(self, val): self._params_ = val

    @property
    def CutFlow(self) -> dict:
        cdef pair[string, int] it
        cdef dict output = {}
        for it in self.sel.cutflow: output[env(it.first)] = it.second
        return output


    @property
    def AverageTime(self): return self.ptr.Mean()

    @property
    def StdevTime(self): return self.ptr.StandardDeviation()

    @property
    def Luminosity(self): return self.ptr.Luminosity()

    @property
    def nPassedEvents(self): return self.sel.selection_weights.size()

    @property
    def TotalEvents(self): return self.sel.all_weights.size()

    @property
    def AllowFailure(self): return self.sel.allow_failure

    @AllowFailure.setter
    def AllowFailure(self, bool val): self.sel.allow_failure = val

    @property
    def hash(self) -> str: return env(self.ptr.Hash())

    @property
    def index(self) -> int: return self.sel.event_index

    @property
    def Tag(self) -> str: return env(self.sel.event_tagging)

    @Tag.setter
    def Tag(self, str val): self.sel.event_tagging = enc(val)

    @property
    def Tree(self) -> str: return env(self.sel.event_tree)

    @property
    def cached(self) -> bool: return self.sel.cached

    @cached.setter
    def cached(self, bool val) -> bool: self.sel.cached = val

    @property
    def ROOT(self) -> str: return env(self.sel.event_root)

    @property
    def Selection(self) -> bool: return self.ptr.is_selection

    @property
    def SelectionName(self) -> str: return env(self.sel.event_name)

#distutils: language = c++
from Event cimport CyEvent
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool
from typing import Union

cdef class EventTemplate:
    cdef CyEvent* ptr
    cdef dict _leaves
    cdef list _Trees
    cdef list _Branches 
    cdef list _Leaves 
    cdef dict _Objects

    def __cinit__(self):
        self.ptr = new CyEvent()
        self._Trees = []
        self._Branches = []
        self._Leaves = []
        self._Objects = {}
        self._leaves = {"event" : {}}
    
    def __init__(self):
        pass

    def __dealloc__(self):
        del self.ptr

    def __hash__(self) -> int:
        return int(self.hash[:8], 0)

    def __eq__(self, other) -> bool:
        if isinstance(self, str): return False
        if isinstance(other, str): return False
        cdef EventTemplate s = self
        cdef EventTemplate o = other 
        return s.hash == o.hash

    @property
    def __interpret__(self) -> dict:
        cdef str i
        cdef list exl = ["CommitHash", "Deprecated"]
        for i, v in zip(self.__dict__, self.__dict__.values()):
            if isinstance(v, list): continue
            if isinstance(v, dict): continue
            if i in exl: continue
            self._leaves["event"][i] = v
        for i in self._Objects:
            if self._Objects[i]._init != True:
                self._Objects[i] = self._Objects[i]()
            self._leaves[i] = self._Objects[i].__interpret__
        
        cdef list col = []
        col += self.Branches
        for i in self._leaves:
            col += list(self._leaves[i].values())
        self.Leaves += col
        self._Objects["event"] = self
        return self._leaves
    
    @__interpret__.setter
    def __interpret__(self, dict val) -> void:
        cdef str i
        for i in val:
            setattr(self, i, val[i])

    def __compiler__(self, inpt: Union[dict]) -> void:
        cdef str i, k, tr
        cdef dict val, _inpt
        cdef EventTemplate ev
        
        for tr in self._Trees:
            ev = self if self._Trees[0] == tr else self.clone
            ev.__interpret__
            ev.Tree = tr
            _inpt = {k.split(tr + "/")[-1] : inpt[k] for k in inpt if tr in k}
            for i in list(ev._Objects):
                val = ev._leaves[i] if i == "event" else ev._Objects[i].__interpret__
                val = {k : _inpt[val[k]] for k in val if val[k] in _inpt}
                if len(val) == 0: 
                    del ev._Objects[i]
                    continue
                
                ev._Objects[i].__interpret__ = val
                if i == "event": continue
                setattr(ev, i, {p : ev._Objects[i].Children[p] for p in range(len(ev._Objects[i].Children))})
                del ev._Objects[i]
            self._Trees[self._Trees.index(tr)] = ev
            ev._leaves = {}
            ev._Leaves = []
            ev._Branches = []
            inpt = {k : inpt[k] for k in inpt if tr not in k}
        del inpt
    
    def CompileEvent(self):
        pass

    @property
    def clone(self):
        v = self.__new__(self.__class__)
        v.__init__()
        return v

    @property 
    def index(self) -> int: return self.ptr.index
    
    @index.setter
    def index(self, val: Union[int, str]):
        if isinstance(val, int): self.ptr.index = val
        elif isinstance(val, str): self._leaves["event"]["index"] = val
   
    @property
    def Tree(self) -> str: return self.ptr.tree.decode("UTF-8")

    @property
    def Trees(self) -> list: return self._Trees
 
    @property
    def Branches(self) -> list: return self._Branches
  
    @property
    def Leaves(self) -> list: return self._Leaves
 
    @property
    def Objects(self) -> dict: return self._Objects

    @Tree.setter
    def Tree(self, str val) -> void: 
        self.ptr.tree = val.encode("UTF-8")

    @Trees.setter
    def Trees(self, val: Union[str, list]) -> void:
        if isinstance(val, str): val = [val]
        self._Trees += val 
   
    @Branches.setter
    def Branches(self, val: Union[str, list]) -> void:
        if isinstance(val, str): val = [val]
        self._Branches += val 
   
    @Leaves.setter
    def Leaves(self, val: Union[str, list]) -> void:
        if isinstance(val, str): val = [val]
        self._Leaves += val 

    @property 
    def hash(self) -> str: return self.ptr.Hash().decode("UTF-8")
    
    @hash.setter
    def hash(self, str val) -> void:
        self.ptr.Hash(val.encode("UTF-8"))

    @property
    def Deprecated(self) -> bool: return self.ptr.deprecated

    @Deprecated.setter
    def Deprecated(self, bool val) -> void:
        self.ptr.deprecated = val

    @property
    def _init(self) -> bool:
        return True

    @Objects.setter
    def Objects(self, val: Union[dict]) -> void:
        self._Objects = val


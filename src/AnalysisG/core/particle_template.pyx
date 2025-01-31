# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.map cimport map, pair
from libcpp.string cimport string
from libcpp.vector cimport vector

from AnalysisG.core.tools cimport *
from AnalysisG.core.structs cimport particle_t
from AnalysisG.core.particle_template cimport particle_template

cdef class ParticleTemplate:

    def __cinit__(self):
        self.children = []
        self.parents = []
        if type(self) is not ParticleTemplate: return
        self.ptr = new particle_template()

    def __init__(self, inpt = None):
        if inpt is None: return
        cdef list keys = [i for i in self.__dir__() if not i.startswith("__")]
        for i in keys:
            try: setattr(self, i, inpt["extra"][i])
            except KeyError: continue
            except AttributeError: continue
        self.ptr.data = <particle_t>(inpt["data"])

    def __dealloc__(self):
        if type(self) is not ParticleTemplate: return
        del self.ptr

    def __reduce__(self): 
        cdef list keys = [i for i in self.__dir__() if not i.startswith("__")]
        cdef dict out = {}
        out["extra"] = {i : getattr(self, i) for i in keys if not callable(getattr(self, i))}
        out["data"]  = self.ptr.data
        return self.__class__, (out,)

    def __hash__(self): return int(string(self.ptr.hash).substr(0, 8), 0)
    def __add__(self, ParticleTemplate other):
        cdef ParticleTemplate p = self.clone()
        p.ptr.iadd(other.ptr)
        p.ptr.iadd(self.ptr)
        if len(other.Children): p.Children += other.Children
        if len(other.Parent):   p.Parent   += other.Parent
        if len(self.Children):  p.Children += self.Children
        if len(self.Parent):    p.Parent   += self.Parent
        return p

    def __radd__(self, other):
        if not self.is_self(other): return self
        return self.__add__(other)

    def __eq__(self, other):
        if not self.is_self(other): return False
        cdef ParticleTemplate o = other
        return self.ptr[0] == o.ptr[0]

    def __getleaves__(self):
        cdef str i
        cdef dict leaves = {}
        cdef pair[string, string] x
        for x in self.ptr.leaves: leaves[env(x.first)] = env(x.second)
        for i, v in zip(self.__dict__, self.__dict__.values()):
            if not isinstance(v, str): continue
            self.ptr.leaves[enc(i)] = enc(v)
            leaves[i] = v
        return leaves

    cpdef ParticleTemplate clone(self):
        v = self.__class__
        v = v()
        v.Type = self.Type
        return v

    def is_self(self, inpt):
        if isinstance(inpt, ParticleTemplate): return True
        return issubclass(inpt.__class__, ParticleTemplate)

    cpdef double DeltaR(self, ParticleTemplate other):
        return self.ptr.DeltaR(other.ptr)

    @property
    def hash(self) -> str: return env(self.ptr.hash)

    @property
    def index(self) -> int: return self.ptr.data.index

    @property
    def px(self) -> double: return self.ptr.px

    @property
    def py(self) -> double: return self.ptr.py

    @property
    def pz(self) -> double: return self.ptr.pz

    @property
    def pt(self) -> double: return self.ptr.pt

    @property
    def eta(self) -> double: return self.ptr.eta

    @property
    def phi(self) -> double: return self.ptr.phi

    @property
    def e(self) -> double: return self.ptr.e

    @property
    def Mass(self) -> double: return self.ptr.mass

    @property
    def charge(self) -> double: return self.ptr.charge

    @property
    def pdgid(self) -> int: return self.ptr.pdgid

    @property
    def symbol(self) -> str : return env(self.ptr.symbol)

    @property
    def Type(self) -> str: return env(self.ptr.data.type)

    @property
    def lepdef(self) -> vector[int]: return self.ptr.data.lepdef

    @property
    def nudef(self) -> vector[int]: return self.ptr.data.nudef

    @property
    def is_lep(self) -> bool: return self.ptr.is_lep

    @property
    def is_nu(self) -> bool: return self.ptr.is_nu

    @property
    def is_b(self) -> bool: return self.ptr.is_b

    @property
    def is_add(self) -> bool: return self.ptr.is_add

    @property
    def LeptonicDecay(self) -> bool: return self.ptr.lep_decay

    @index.setter
    def index(self, val):
        try: self.ptr.data.index = val
        except TypeError:
            try: self.ptr.add_leaf(b'index', enc(val))
            except: self.__dict__["index"] = val

    @px.setter
    def px(self, val):
        try: self.ptr.px = val
        except TypeError: self.ptr.add_leaf(b'px', enc(val))

    @py.setter
    def py(self, val):
        try: self.ptr.py = val
        except TypeError: self.ptr.add_leaf(b'py', enc(val))

    @pz.setter
    def pz(self, val):
        try: self.ptr.pz = val
        except TypeError: self.ptr.add_leaf(b'pz', enc(val))

    @pt.setter
    def pt(self, val):
        try: self.ptr.pt = val
        except TypeError: self.ptr.add_leaf(b'pt', enc(val))

    @eta.setter
    def eta(self, val):
        try: self.ptr.eta = val
        except TypeError: self.ptr.add_leaf(b'eta', enc(val))

    @phi.setter
    def phi(self, val):
        try: self.ptr.phi = val
        except TypeError: self.ptr.add_leaf(b'phi', enc(val))

    @e.setter
    def e(self, val):
        try: self.ptr.e = val
        except TypeError: self.ptr.add_leaf(b'e', enc(val))

    @Mass.setter
    def Mass(self, val):
        try: self.ptr.mass = val
        except TypeError: self.ptr.add_leaf(b'Mass', enc(val))

    @charge.setter
    def charge(self, val):
        try: self.ptr.charge = val
        except TypeError: self.ptr.add_leaf(b'charge', enc(val))

    @pdgid.setter
    def pdgid(self, val):
        try: self.ptr.pdgid = val
        except TypeError: self.ptr.add_leaf(b'pdgid', enc(val))

    @symbol.setter
    def symbol(self, str val): self.ptr.symbol = enc(val)

    @Type.setter
    def Type(self, str val): self.ptr.data.type = enc(val)

    @lepdef.setter
    def lepdef(self, vector[int] val): self.ptr.data.lepdef = val

    @nudef.setter
    def nudef(self, vector[int] val): self.ptr.data.nudef = val

    @property
    def Children(self) -> list:
        self.children = list(set(self.children))
        return self.children

    @Children.setter
    def Children(self, inpt):
        cdef ParticleTemplate p
        cdef particle_template* ptx

        if not len(inpt):
            self.ptr.children.clear()
            self.children = []
            return

        if isinstance(inpt, list):
            for p in inpt:
                ptx = p.ptr
                if not self.ptr.register_child(ptx): continue
                self.children.append(p)
        else: self.Children += [inpt]

    @property
    def Parent(self) -> list:
        self.parents = list(set(self.parents))
        return self.parents

    @Parent.setter
    def Parent(self, inpt):
        cdef ParticleTemplate p
        cdef particle_template* ptx

        if not len(inpt):
            self.ptr.parents.clear();
            self.parents = []

        if isinstance(inpt, list):
            for p in inpt:
                ptx = p.ptr
                if not self.ptr.register_parent(ptx): continue
                self.parents.append(p)
        else: self.Parent += [inpt]




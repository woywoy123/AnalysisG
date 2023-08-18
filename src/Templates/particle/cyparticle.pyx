# distuils: language = c++
# cython: language_level = 3
from cyparticle cimport CyParticleTemplate, ExportParticleTemplate
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool
from typing import Union

cdef string enc(str val): return val.encode("UTF-8")
cdef str env(string val): return val.decode("UTF-8")

cdef class ParticleTemplate:

    cdef CyParticleTemplate* ptr
    cdef list Children
    cdef list Parent

    def __cinit__(self, double px = 0, double py = 0, double pz = 0, double e = -1):
        self.ptr = new CyParticleTemplate()
        self.Children = []
        self.Parent = []

    def __init__(self):
        pass

    def __dealloc__(self):
        del self.ptr

    def __add__(self, ParticleTemplate other) -> ParticleTemplate:
        cdef ParticleTemplate _p = self.clone()
        _p.ptr.iadd(self.ptr[0] + other.ptr)
        _p.Children = list(set(other.Children + self.Children))
        _p.Parent = list(set(other.Parent + self.Parent))
        return _p

    def __radd__(self, other) -> ParticleTemplate:
        if not self.is_self(other): return self
        return self.__add__(other)

    def __iadd__(self, ParticleTemplate other) -> ParticleTemplate:
        self.Children = list(set(other.Children + self.Children))
        self.Parent = list(set(other.Parent + self.Parent))
        self.ptr.iadd(other.ptr)
        return self

    def __getstate__(self) -> dict:
        cdef ExportParticleTemplate x = self.ptr.MakeMapping()
        cdef str key
        cdef dict out = {"__Export__" : x}
        for key in self.__dict__:
            out[key] = self.__dict__[key]
        return out

    def __hash__(self):
        return int(self.hash[:8], 0)

    def __eq__(self, other) -> bool:
        if not self.is_self(other): return False
        cdef ParticleTemplate o = other
        return self.ptr == o.ptr


    def __getleaves__(self):
        cdef str i
        cdef pair[string, string] x
        cdef dict leaves = {}
        for x in self.ptr.leaves: leaves[env(x.first)] = env(x.second)
        for i, v in zip(self.__dict__, self.__dict__.values()):
            if not isinstance(v, str): continue
            leaves[i] = v
        return leaves


    def __build__(self, dict variables):
        cdef ParticleTemplate p
        cdef dict inpt = {}
        cdef dict x = {}
        cdef bool get
        cdef str k

        for k in variables:
            try: inpt[k] = variables[k].tolist()
            except AttributeError: inpt[k] = variables[k]

        while True: 
            x = {}
            get = False
            for k in list(inpt):
                try: inpt[k] = inpt[k].tolist()
                except AttributeError: pass

                try: x[k] = inpt[k].pop()
                except AttributeError: x[k] = inpt[k]
                except IndexError: pass

                if k not in x: continue

                try: x[k] = x[k].item()
                except AttributeError: pass

                try: len(x[k])
                except TypeError: get = True

            if len(x) == 0: break
            if not get: self.__build__(x)
            else:
                p = self.clone()
                for k in x: setattr(p, k, x[k])
                self.Children.append(p)

    def clone(self) -> ParticleTemplate:
        v = self.__new__(self.__class__)
        v.__init__()
        v.Type = self.Type
        return v

    def is_self(self, inpt) -> bool:
        if isinstance(inpt, ParticleTemplate): return True
        return issubclass(inpt.__class__, ParticleTemplate)

    @property
    def hash(self) -> str: return env(self.ptr.hash())

    @property
    def index(self) -> int: return self.ptr.index

    @property
    def px(self) -> double: return self.ptr.px()

    @property
    def py(self) -> double: return self.ptr.py()

    @property
    def pz(self) -> double: return self.ptr.pz()

    @property
    def pt(self) -> double: return self.ptr.pt()

    @property
    def eta(self) -> double: return self.ptr.eta()

    @property
    def phi(self) -> double: return self.ptr.phi()

    @property
    def e(self) -> double: return self.ptr.e()

    @property
    def Mass(self) -> double: return self.ptr.mass()

    @property
    def charge(self) -> double: return self.ptr.charge()

    @property
    def pdgid(self) -> int: return self.ptr.pdgid()

    @property
    def symbol(self) -> str : return env(self.ptr.symbol())

    @property
    def Type(self) -> str: return env(self.ptr.type)

    @property
    def lepdef(self) -> vector[int]: return self.ptr.lepdef

    @property
    def nudef(self) -> vector[int]: return self.ptr.nudef

    @property
    def is_lep(self) -> bool: return self.ptr.is_lep()

    @property
    def is_nu(self) -> bool: return self.ptr.is_nu()

    @property
    def is_b(self) -> bool: return self.ptr.is_b()

    @property
    def is_add(self) -> bool: return self.ptr.is_add()

    @property
    def LeptonicDecay(self) -> bool: return self.ptr.lep_decay()

    @index.setter
    def index(self, val: Union[str, int, float]):
        try: self.ptr.index = val
        except TypeError: self.ptr.addleaf(b'index', enc(val))

    @px.setter
    def px(self, val: Union[str, double]):
        try: self.ptr.px(val)
        except TypeError: self.ptr.addleaf(b'px', enc(val))

    @py.setter
    def py(self, val: Union[str, double]):
        try: self.ptr.py(val)
        except TypeError: self.ptr.addleaf(b'py', enc(val))

    @pz.setter
    def pz(self, val: Union[str, double]):
        try: self.ptr.pz(val)
        except TypeError: self.ptr.addleaf(b'pz', enc(val))

    @pt.setter
    def pt(self, val: Union[str, double]):
        try: self.ptr.pt(val)
        except TypeError: self.ptr.addleaf(b'pt', enc(val))

    @eta.setter
    def eta(self, val: Union[str, double]):
        try: self.ptr.eta(val)
        except TypeError: self.ptr.addleaf(b'eta', enc(val))

    @phi.setter
    def phi(self, val: Union[str, double]):
        try: self.ptr.phi(val)
        except TypeError: self.ptr.addleaf(b'phi', enc(val))

    @e.setter
    def e(self, val: Union[str, double]):
        try: self.ptr.e(val)
        except TypeError: self.ptr.addleaf(b'e', enc(val))

    @Mass.setter
    def Mass(self, val: Union[str, double]):
        try: self.ptr.mass(val)
        except TypeError: self.ptr.addleaf(b'Mass', enc(val))

    @charge.setter
    def charge(self, val: Union[str, double]):
        try: self.ptr.charge(val)
        except TypeError: self.ptr.addleaf(b'charge', enc(val))

    @pdgid.setter
    def pdgid(self, val: Union[str, double, int]):
        try: self.ptr.pdgid(val)
        except TypeError: self.ptr.addleaf(b'pdgid', enc(val))

    @symbol.setter
    def symbol(self, str val):
        self.ptr.symbol(enc(val))

    @Type.setter
    def Type(self, str val):
        self.ptr.type = enc(val)

    @lepdef.setter
    def lepdef(self, vector[int] val):
        self.ptr.lepdef = val

    @nudef.setter
    def nudef(self, vector[int] val):
        self.ptr.nudef = val

    @property
    def Children(self) -> list:
        self.Children = list(set(self.Children))
        return self.Children

    @Children.setter
    def Children(self, val: Union[list, ParticleTemplate]):
        if self.is_self(val): self.Children = [val]
        else: self.Children = val
        self.Children = list(set(self.Children))

        cdef ParticleTemplate x
        self.ptr.children.clear()
        for x in self.Children: self.ptr.children[x.ptr.hash()] = x.ptr

    @property
    def Parent(self) -> list:
        self.Parent = list(set(self.Parent))
        return self.Parent

    @Parent.setter
    def Parent(self, val: Union[list, ParticleTemplate]):
        if self.is_self(val): self.Parent = [val]
        else: self.Parent = val
        self.Parent = list(set(self.Parent))

        cdef ParticleTemplate x
        self.ptr.parent.clear()
        for x in self.Parent: self.ptr.parent[x.ptr.hash()] = x.ptr







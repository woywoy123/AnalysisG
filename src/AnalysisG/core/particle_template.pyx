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
        self.is_owner = True
        self.keys = []
        if type(self) is not ParticleTemplate: return
        self.ptr = new particle_template()

    def __init__(self, inpt = None):
        if inpt is None: return
        if not len(self.keys): self.keys = [i for i in self.__dir__() if not i.startswith("__")]
        for i in self.keys:
            try: setattr(self, i, inpt[b"extra"][i])
            except KeyError: continue
            except AttributeError: continue
        self.ptr.data = <particle_t>(inpt[b"data"])

    def __dealloc__(self):
        if type(self) is not ParticleTemplate: return
        if not self.is_owner: return
        del self.ptr

    def __reduce__(self, dict dictout = {}):
        cdef map[string, map[string, particle_t]] mx
        cdef pair[string, map[string, particle_t]] itx

        try:
            dictout[self.ptr.hash]
            self.ptr._is_serial = True
        except KeyError:
            self.ptr._is_serial = False
            mx = self.ptr.__reduce__()
            for itx in mx: 
                try: dictout[itx.first] |= dict(itx.second)
                except KeyError: dictout[itx.first] = itx.second
            self.ptr._is_serial = True

        if not len(self.keys): self.keys = [i for i in self.__dir__() if not i.startswith("__")]
        cdef dict slf_dict = dict(dictout[self.ptr.hash])

        try: return slf_dict.pop(b"__class__"), (slf_dict,)
        except KeyError: pass

        dictout[self.ptr.hash][b"extra"] = {i : getattr(self, i) for i in self.keys if not callable(getattr(self, i))}
        dictout[self.ptr.hash][b"__class__"] = self.__class__
        return self.__reduce__(dictout)

    def __hash__(self): return int(string(self.ptr.hash).substr(0, 8), 0)

    def __add__(self, ParticleTemplate other):
        cdef ParticleTemplate p = self.clone()
        p.ptr.iadd(other.ptr)
        p.ptr.iadd(self.ptr)
        if len(other.Children): p.Children += other.Children
        if len(other.Parents):  p.Parents  += other.Parents
        if len(self.Children):  p.Children += self.Children
        if len(self.Parents):   p.Parents  += self.Parents
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

    cdef void set_particle(self, particle_template* ox):
        del self.ptr
        self.ptr = ox
        self.is_owner = False

    cdef list make_particle(self, map[string, particle_template*] px):
        cdef list out = []
        cdef ParticleTemplate pi
        cdef pair[string, particle_template*] p
        for p in px:
            pi = ParticleTemplate()
            pi.set_particle(p.second)
            out.append(pi)
        return out

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
        if len(self.children): return list(set(self.children))
        self.children = self.children + self.make_particle(self.ptr.children)
        return list(set(self.children))

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
    def Parents(self) -> list:
        if len(self.parents): return list(set(self.parents))
        self.parents = self.parents + self.make_particle(self.ptr.parents)
        return list(set(self.parents))

    @Parents.setter
    def Parents(self, inpt):
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
        else: self.Parents += [inpt]




#distutils: language = c++
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from typing import Union
from Particles cimport CyParticle

def proxy(obj, arg):
    elif isinstance(arg, str): obj(<string>arg.encode("UTF-8"))

cdef class ParticleTemplate:
    cdef CyParticle* ptr
    cdef public list Children
    cdef public list Parent
    cdef dict _leafs

    def __cinit__(self):
        self.ptr = new CyParticle()
        self._leafs = {}
    
    def __init__(self):
        self.Children = []
        self.Parent = []
  
    @property
    def __interpret__(self):
        cdef str i, v
        for i, v in zip(self.__dict__, self.__dict__.values()):
            self._leafs[i] = v
        return self._leafs

    def __dealloc__(self):
        del self.ptr
 
    def __add__(self, other) -> ParticleTemplate:
        if isinstance(self, int): return other
        if isinstance(other, int): return self

        cdef ParticleTemplate p = ParticleTemplate()
        cdef ParticleTemplate s = self
        cdef ParticleTemplate o = other 

        s.ptr._UpdateState()
        o.ptr._UpdateState()
        p.ptr[0] = s.ptr[0] + o.ptr[0]
        return p
    
    def __eq__(self, other) -> bool:
        cdef ParticleTemplate s = self
        cdef ParticleTemplate o = other 
        s.ptr._UpdateState()
        o.ptr._UpdateState()
        return s.ptr[0] == o.ptr[0]
    
    def __hash__(self) -> int:
        return int(self.hash[:8], 0)

    def __str__(self) -> str:
        cdef str i = "============\n"
        if self.pdgid != 0:
            i += " pdgid: " + str(self.pdgid) + "\n"
            i += "Symbol: " + str(self.symbol) + "\n"
        i += "    pt: " + str(self.pt) + "\n"
        i += "   eta: " + str(self.eta) + "\n"
        i += "   phi: " + str(self.phi) + "\n"
        i += "energy: " + str(self.e) + "\n"
        return i
    
    @property 
    def index(self) -> int:
        return self.ptr.index
    
    @index.setter
    def index(self, val: Union[int]) -> void: 
        self.index(val)
    
    @property
    def hash(self) -> str:
        self.ptr._UpdateState()
        return self.ptr.Hash().decode("UTF-8")
    
    @property
    def px(self) -> double:
        self.ptr._UpdateState()
        return self.ptr.px()
    
    @px.setter
    def px(self, val: Union[str, float]) -> void:
        if isinstance(val, float): self.ptr.px(<double> arg)
        elif isinstance(val, str): self._leafs["px"] = val

    @property
    def py(self) -> double:
        self.ptr._UpdateState()
        return self.ptr.py()

    @py.setter
    def py(self, val: Union[str, float]) -> void:
        if isinstance(val, float): self.ptr.py(<double> arg)
        elif isinstance(val, str): self._leafs["py"] = val

    @property
    def pz(self) -> double:
        self.ptr._UpdateState()
        return self.ptr.pz()

    @pz.setter
    def pz(self, double val) -> void:
        self.ptr.pz(val)

    @property
    def pt(self) -> double:
        self.ptr._UpdateState()
        return self.ptr.pt()

    @pt.setter
    def pt(self, double val) -> void:
        self.ptr.pt(val)

    @property
    def eta(self) -> double:
        self.ptr._UpdateState()
        return self.ptr.eta()

    @eta.setter
    def eta(self, double val) -> void:
        self.ptr.eta(val)

    @property
    def phi(self) -> double:
        self.ptr._UpdateState()
        return self.ptr.phi()

    @phi.setter
    def phi(self, double val) -> void:
        self.ptr.phi(val)
    
    @property
    def e(self) -> double:
        self.ptr._UpdateState()
        return self.ptr.e()

    @e.setter 
    def e(self, double val) -> void:
        self.ptr.e(val)
    
    @property
    def Mass(self) -> double:
        return self.ptr.Mass()
    
    @Mass.setter
    def Mass(self, double val) -> void:
        self.ptr.Mass(val)
    
    def DeltaR(ParticleTemplate self, ParticleTemplate other) -> double:
        return self.ptr.DeltaR(other.ptr[0])
    
    @property
    def pdgid(self) -> int:
        return self.ptr.pdgid()

    @pdgid.setter
    def pdgid(self, int val) -> void:
        self.ptr.pdgid(val)

    @property
    def charge(self) -> int:
        return self.ptr.charge()

    @charge.setter
    def charge(self, int val) -> void:
        self.ptr.charge(val)
    
    @property
    def symbol(self) -> str:
        return self.ptr.symbol().decode("UTF-8")
    
    @property
    def is_lep(self) -> bool:
        return self.ptr.is_lep()

    @property
    def is_nu(self) -> bool:
        return self.ptr.is_nu()

    @property 
    def is_b(self) -> bool:
        return self.ptr.is_b()

    @property
    def is_add(self) -> bool:
        return not (self.is_lep or self.is_nu or self.is_b)
    
    @property
    def lepdef(self) -> vector[int]:
        return self.ptr._lepdef 

    @property
    def nudef(self) -> vector[int]:
        return self.ptr._nudef 

    @lepdef.setter
    def lepdef(self, vector[signed int] val) -> void:
        self.ptr._lepdef = val

    @nudef.setter
    def nudef(self, vector[signed int] val) -> void:
        self.ptr._nudef = val
    
    ## ___ PROTECTION AGAINST STRING NAMESPACE OF SETTER ____ # 
    #cdef void index(self, str v): self._leafs["index"] = v
    #cdef void hash(self, str v): self._leafs["hash"] = v
    #cdef void px(self, str v): self._leafs["px"] = v       
    #cdef void py(self, str v): self._leafs["py"] = v
    #cdef void pz(self, str v): self._leafs["pz"] = v
    #cdef void pt(self, str v): self._leafs["pt"] = v
    #cdef void eta(self, str v): self._leafs["eta"] = v
    #cdef void phi(self, str v): self._leafs["phi"] = v
    #cdef void e(self, str v): self._leafs["e"] = v
    #cdef void Mass(self, str v): self._leafs["Mass"] = v
    #cdef void pdgid(self, str v): self._leafs["pdgid"] = v
    #cdef void charge(self, str v): self._leafs["charge"] = v

    ## ___ C-TYPE INPUT ___ #
    #cdef void index(self, int val): self.ptr.index = val
    #cdef void px(self, double val): self.ptr.px(val)
    #cdef void py(self, double val): self.ptr.py(val)
    #cdef void pz(self, double val): self.ptr.pz(val)
    #cdef void pt(self, double val): self.ptr.pt(val)
    #cdef void eta(self, double val): self.ptr.eta(val)
    #cdef void phi(self, double val): self.ptr.phi(val)
    #cdef void e(self, double val): self.ptr.e(val)
    #cdef void Mass(self, double val): self.ptr.Mass(val)
    #cdef void pdgid(self, int val): self.ptr.pdgid(val)
    #cdef void charge(self, int val): self.ptr.charge(val)
   

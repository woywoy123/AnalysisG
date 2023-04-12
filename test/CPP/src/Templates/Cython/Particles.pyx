#distutils: language = c++
from Particles cimport CyParticle

cdef class ParticleTemplate:
    cdef CyParticle* ptr

    def __cinit__(self):
        self.ptr = new CyParticle()

    def __dealloc__(self):
        del self.ptr
 
    def __add__(self, other):
        if isinstance(self, int): return other
        if isinstance(other, int): return self

        cdef ParticleTemplate p = ParticleTemplate()
        cdef ParticleTemplate s = self
        cdef ParticleTemplate o = other 

        s.ptr._UpdateState()
        o.ptr._UpdateState()
        p.ptr[0] = s.ptr[0] + o.ptr[0]
        return p
    
    def __eq__(self, other):
        cdef ParticleTemplate s = self
        cdef ParticleTemplate o = other 
        s.ptr._UpdateState()
        o.ptr._UpdateState()
        return s.ptr[0] == o.ptr[0]
    
    def __hash__(self):
        return int(self.hash[:8], 0)
    
    @property 
    def index(self):
        return self.ptr.index
    
    @index.setter
    def index(self, int val):
        self.ptr.index = val
    
    @property
    def hash(self):
        self.ptr._UpdateState()
        return self.ptr.Hash().decode("UTF-8")
    
    @property
    def px(self):
        self.ptr._UpdateState()
        return self.ptr.px()

    @px.setter
    def px(self, double val):
        self.ptr.px(val)

    @property
    def py(self):
        self.ptr._UpdateState()
        return self.ptr.py()

    @py.setter
    def py(self, double val):
        self.ptr.py(val)

    @property
    def pz(self):
        self.ptr._UpdateState()
        return self.ptr.pz()

    @pz.setter
    def pz(self, double val):
        self.ptr.pz(val)

    @property
    def pt(self):
        self.ptr._UpdateState()
        return self.ptr.pt()

    @pt.setter
    def pt(self, double val):
        self.ptr.pt(val)

    @property
    def eta(self):
        self.ptr._UpdateState()
        return self.ptr.eta()

    @eta.setter
    def eta(self, double val):
        self.ptr.eta(val)

    @property
    def phi(self):
        self.ptr._UpdateState()
        return self.ptr.phi()

    @phi.setter
    def phi(self, double val):
        self.ptr.phi(val)
    
    @property
    def e(self):
        self.ptr._UpdateState()
        return self.ptr.e()

    @e.setter
    def e(self, double val):
        self.ptr.e(val)
    
    @property
    def Mass(self):
        return self.ptr.Mass()
    
    @Mass.setter
    def Mass(self, double val):
        self.ptr.Mass(val)
    
    def DeltaR(ParticleTemplate self, ParticleTemplate other):
        return self.ptr.DeltaR(other.ptr[0])

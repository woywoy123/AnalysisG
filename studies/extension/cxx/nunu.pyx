cimport nunu

cdef class Particle: 
    def __init__(self): pass
    def __dealloc__(self): del self.ptr

    @property
    def p(self):      return self.ptr.p()

    @property
    def p2(self):     return self.ptr.p2()   

    @property
    def m(self):      return self.ptr.m()    

    @property
    def m2(self):     return self.ptr.m2()   

    @property
    def beta(self):   return self.ptr.beta() 

    @property
    def beta2(self):  return self.ptr.beta2()

    @property
    def phi(self):    return self.ptr.phi()  

    @property
    def theta(self):  return self.ptr.theta()

    @property
    def px(self):     return self.ptr.px     

    @property
    def py(self):     return self.ptr.py     

    @property
    def pz(self):     return self.ptr.pz     

    @property
    def e (self):     return self.ptr.e      

    @property
    def d(self):     return self.ptr.d 



cdef class NuNu:
    def __init__(self, 
            b1_px, b1_py, b1_pz, b1_e, l1_px, l1_py, l1_pz, l1_e, 
            b2_px, b2_py, b2_pz, b2_e, l2_px, l2_py, l2_pz, l2_e, 
            mt1  , mt2  , mw1  , mw2):

        self.nu = new nunu(
            b1_px, b1_py, b1_pz, b1_e, l1_px, l1_py, l1_pz, l1_e, 
            b2_px, b2_py, b2_pz, b2_e, l2_px, l2_py, l2_pz, l2_e, 
            mt1  , mt2  , mw1  , mw2)

    def __dealloc__(self): del self.nu

    def misc(self): self.nu.get_misc()

    def generate(self, metx, mety, metz): 
        cdef int lx = self.nu.generate(metx, mety, metz)
        cdef list out = [[None, None] for _ in range(lx)]
        cdef Particle p1; 
        cdef Particle p2; 
        cdef particle* p1_; 
        cdef particle* p2_; 

        for x in range(lx):
            p1 = Particle()
            p2 = Particle()
            p1_ = p1.ptr
            p2_ = p2.ptr
            self.nu.get_nu(&p1_, &p2_, x)
            out[x][0] = p1; out[x][1] = p2
        self.nu._clear()
        return out
        



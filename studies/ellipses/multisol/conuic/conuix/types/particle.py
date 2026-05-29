from conuix.base.atomics import *
import math 

class Particle:
    def __init__(self, px = None, py = None, pz = None, e = None):
        self.px = px; self.py = py; self.pz = pz; self.e = e; 
        self.skp = self.mass2 < 0
        self.e = np.sqrt(self.mass2 + self.p2)


    def __str__(self):
        o  =       string(self, "px")   + " " + string(self, "py")
        o += " " + string(self, "pz")   + " " + string(self, "e")
        o += " " + string(self, "mass")
        return o
    def __add__(self, o):
        if isinstance(o, str): return o + self.__str__()
        return Particle(o.px + self.px, o.py + self.py, o.pz + self.pz, o.e  + self.e)

    def __radd__(self, o):
        if o != 0: return self.__add__(o)
        return Particle(self.px, self.py, self.pz, self.e)

    @property           
    def mass2(self): return self.e**2 - (self.px**2 + self.py**2 + self.pz**2)

    @property
    def mass(self): return self.mass2**0.5

    @property
    def p2(self): return (self.px**2 + self.py**2 + self.pz**2)

    @property
    def p(self): return self.p2**0.5

    @property
    def beta2(self): return 1 - (self.mass / self.e)**2

    @property
    def beta(self): return (complex(self.beta2)**0.5).real

    @property
    def phi(self): return math.atan2(self.py, self.px)

    @property
    def theta(self): return math.atan2((self.px**2 + self.py**2)**0.5, self.pz)

    @property
    def pT(self): return (self.px ** 2 + self.py ** 2) ** 0.5



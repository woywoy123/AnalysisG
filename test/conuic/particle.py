from atomics import *

class Particle:
    def __init__(self, px = None, py = None, pz = None, e = None, hx = None):
        self.px = px; self.py = py; 
        self.pz = pz; self.e = e; 
        self.hash = hx
        self.top_index = -1

    def __str__(self):
        o  =       string(self, "px")   + " " + string(self, "py")
        o += " " + string(self, "pz")   + " " + string(self, "e")
        o += " " + string(self, "hash") + " " + string(self, "top_index")
        return o

    @property
    def mass2(self): return abs(self.e**2 - (self.px**2 + self.py**2 + self.pz**2))

    @property
    def mass(self): return self.mass2**0.5

    @property
    def p2(self): return (self.px**2 + self.py**2 + self.pz**2)

    @property
    def p(self): return self.p2**0.5
   
    @property
    def b2(self): return self.p2 / (self.e**2)

    @property
    def b(self): return self.b2**0.5

    @property
    def gev(self): 
        self.px *= 0.001
        self.py *= 0.001
        self.pz *= 0.001
        self.e  *= 0.001



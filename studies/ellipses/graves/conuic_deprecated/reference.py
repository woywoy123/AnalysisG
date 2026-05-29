from particle import *
from atomics import *
import numpy as np
import math

class NuSol(object):
    def __init__(self, b, mu, mW2, mT2, mN2 = 0):
        self._b = b
        self._mu = mu
        self.mW2 = mW2*mW2
        self.mN2 = mN2*mN2
        self.mT2 = mT2*mT2
        self._sx = None
        self._sy = None

    @property
    def b(self): return self._b

    @property
    def mu(self): return self._mu

    @property
    def c(self): return costheta(self._b, self._mu)

    @property
    def s(self): return (1 - self.c**2)**0.5

    @property
    def x0p(self):
        m2 = self.b.mass**2
        return -(self.mT2 - self.mW2 - m2)/(2*self.b.e)

    @property
    def x0(self):
        m2 = self.mu.mass**2
        return -(self.mW2 - m2 - self.mN2)/(2*self.mu.e)

    @property
    def Sx(self):
        if self._sx is not None: return self._sx
        P = self.mu.p
        beta = self.mu.b
        return (self.x0 * beta - P * (1 - beta**2))/beta**2

    @property
    def Sy(self):
        if self._sy is not None: return self._sy
        beta = self.b.b
        return ((self.x0p / beta) - self.c * self.Sx) / self.s

    @property
    def w(self):
        beta_m, beta_b = self.mu.b, self.b.b
        return (beta_m/beta_b - self.c)/self.s

    @property
    def Om2(self): return self.w**2 + 1 - self.mu.b**2

    @property
    def eps2(self):
        return (self.mW2 - self.mN2) * (1 - self.mu.b**2)

    @property
    def x1(self):
        return self.Sx - ( self.Sx + self.w * self.Sy)/self.Om2

    @property
    def y1(self):
        return self.Sy - ( self.Sx + self.w * self.Sy)*self.w/self.Om2

    @property
    def Z2(self):
        p1 = (self.x1**2)* self.Om2
        p2 = - (self.Sy - self.w * self.Sx)**2
        p3 = - (self.mW2 - self.x0**2 - self.eps2)
        return  p1 + p2 + p3

    @property
    def Z(self): return math.sqrt(max(0, self.Z2))

    @property
    def H_tilde(self):
        return np.array([
            [self.Z/math.sqrt(self.Om2)           , 0   , self.x1 - self.mu.p],
            [self.w * self.Z / math.sqrt(self.Om2), 0   , self.y1            ],
            [0,                                   self.Z, 0                  ]])



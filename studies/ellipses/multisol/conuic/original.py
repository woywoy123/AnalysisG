from conuix.types.particle import *
from conuix.base.atomics import * 

from scipy.optimize import leastsq
import numpy as np
import math


class NuSol(object):
    def __init__(self, b, mu, nu, mins = False):
        self._b = b
        self._mu = mu

        wbs = mu + nu
        top = wbs + b

        self.mW2 = wbs.mass ** 2 
        self.mT2 = top.mass ** 2 
        self.mN2 = nu.mass ** 2  
        self.use_minus = mins
        self.truth = nu

    @property
    def b(self): return self._b

    @property
    def mu(self): return self._mu

    @property
    def c(self): return costheta(self._b, self._mu)

    @property
    def s(self): return (1 - self.c**2)**0.5

    @property
    def x0p(self): return -(self.mT2 - self.mW2 - self.b.mass**2)/(2*self.b.e)

    @property
    def x0(self): return -(self.mW2 - self.mu.mass**2 - self.mN2)/(2*self.mu.e)

    @property
    def Sx(self):
        return (self.x0 * self.mu.beta - self.mu.beta * self.mu.e * (1 - self.mu.beta**2))/(self.mu.beta**2)

    @property
    def Sy(self): return ((self.x0p / self.b.beta) - self.c * self.Sx) / self.s

    @property
    def w(self):
        beta_m, beta_b = self.mu.beta, self.b.beta
        return (beta_m/beta_b - self.c)/self.s if not self.use_minus else self.wm

    @property
    def wm(self):
        beta_m, beta_b = self.mu.beta, self.b.beta
        return -(beta_m/beta_b + self.c)/self.s

    @property
    def Om2(self): return self.w**2 + 1 - self.mu.beta**2

    @property
    def eps2(self):
        return (self.mW2 - self.mN2) * (1 - self.mu.beta**2)

    @property
    def x1(self):
        return self.Sx - ( self.Sx + self.w * self.Sy)/self.Om2

    @property
    def y1(self):
        return self.Sy - ( self.Sx + self.w * self.Sy)*self.w / self.Om2

    @property
    def Z2(self):
        p1 =  (self.x1**2)* self.Om2
        p2 = - (self.Sy - self.w * self.Sx)**2
        p3 = - (self.mW2 - self.x0**2 - self.eps2)
        return  p1 + p2 + p3

    @property
    def Z(self):
        z2 = self.Z2
        #if z2 < 0: self.use_minus = True; z2 = self.Z2
        #self.use_minus = False
        return math.sqrt(z2) if z2 > 0 else math.sqrt(-z2)

    @property
    def H_tilde(self):
        return np.array([
            [self.Z/math.sqrt(self.Om2)           , 0   , self.x1 - self.mu.p],
            [self.w * self.Z / math.sqrt(self.Om2), 0   , self.y1            ],
            [0,                                   self.Z, 0                  ]])

    @property
    def R_T(self):
        b_xyz = self.b.px, self.b.py, self.b.pz
        R_z = R(2, -self.mu.phi)
        R_y = R(1, 0.5 * math.pi - self.mu.theta)
        R_x = next(R(0, -math.atan2(z, y)) for x, y, z in (R_y.dot(R_z.dot(b_xyz)),))
        return R_z.T.dot(R_y.T.dot(R_x.T))

    @property
    def H(self): return self.R_T.dot(self.H_tilde)

    @property
    def solution(self):
        H = self.H_tilde
        RT = self.R_T
        tru = np.array([self.truth.px, self.truth.py, self.truth.pz, self.truth.e])
        def nus(ts):
            v = RT.dot(H.dot([math.cos(ts[0]), math.sin(ts[0]), 1]))
            e = sum(v**2 + self.truth.mass**2)**0.5
            return np.array([v[0], v[1], v[2], e])

        def res(par): return sum( (nus(par) - tru) ** 2)

        ts = None
        for i in range(1000):
            if ts is not None: s = ts
            else: s = [0]
            ts, s = leastsq(res, s, ftol = 1e-18, epsfcn = 0.000000001)
            s = res(ts)
            if s > 10: continue
            s = True
            break

        if s is not True: print("!!!!!!!! WARNING NOT CONVERGED")

        return {
                "res"     : res(ts),
                "H_T"     : H, 
                "H"       : self.H, 
                "neutrino": H.dot([math.cos(ts[0]), math.sin(ts[0]), 1]), 
                "labnu"   : RT.dot(H.dot([math.cos(ts[0]), math.sin(ts[0]), 1])), 
                "angle"   : ts[0]
        }




from particle import *
from visualize import *
from atomics import *
from cache import *
from debug import *

import numpy as np
import math

class conuic(matrix, debug):

    def __init__(self, lep, bqrk, event_t = None, runtime = None):
        matrix.__init__(self)
        debug.__init__(self)
        self.ax = runtime.ax

        self.m_nu = 0 

        self.z     = 1000 # scaling factor Z
        self.tau   = None # hyperbolic variable
        self.lstar = None # lambda 
        self.error = None 
        
        self.lep = lep
        self.jet = bqrk
        self.cache()
        self.truth_pair = []
        if event_t is None: return 

        self.is_truth = False
        if self.jet.top_index != self.lep.top_index: return self.test(1)
        if self.lep.top_index not in event_t.truth_pairs: return self.test(1)
        self.truth_pair = event_t.truth_pairs[self.jet.top_index]
        self.is_truth = True
        self.test(1)
    
    def Z2(self, sx, sy):
        out  = self._Z2.a * sx * sx
        out += self._Z2.b * sx * sy
        out += self._Z2.c * sy * sy
        out += self._Z2.d * sx
        out += self._Z2.e 
        return out

    # ---- plus branch 
    def SxP(self, st, ct, z):
        return   z * (self._Sx.a * ct - self._Sx.b * st) + self._Sx.c

    def SyP(self, st, ct, z):
        return   z * (self._Sy.a * ct + self._Sy.b * st) + self._Sy.c

    # ---- minus branch 
    def SxM(self, st, ct, z):
        return - z * (self._Sx.a * ct - self._Sx.b * st) + self._Sx.c

    def SyM(self, st, ct, z):
        return - z * (self._Sy.a * ct + self._Sy.b * st) + self._Sy.c

    # p_mu - branch * Z / Omega  [beta_mu cos(psi) cosh(tau) + Omega * sin(psi) sinh(tau)]
    def x1(self, st, ct, z, b = 1):
        return self._x1.a - b * self._x1.b * z * (self._x1.c * ct + self._x1.d * st)

    # branch * Z / Omega  [Omega cos(psi) sinh(tau) - beta_mu * sin(psi) cosh(tau)]
    def y1(self, st, ct, z, b = 1):
        return              b * self._y1.a * z * (self._y1.b * st - self._y1.c * ct)

    def get_tauZ(self, sx, sy):
        f  = self._tZ.a
        kx = self._tZ.b
        ky = self._tZ.c
        a = (sy + ky) * self.cpsi - (sx + kx) * self.spsi 
        b = (sx + kx) * self.cpsi + (sy + ky) * self.spsi
        t = f * (a / b)

        o = coef_t()
        o.a = t
        o.b = math.atanh(t)
        o.c = f * self.cpsi * cosh(o.b)
        o.c = (sx + kx) / (o.c - self.spsi * sinh(o.b))
        o.d = -1 if o.c < 0 else 1
        o.c = abs(o.c)
        return o

    def masses(self, st, ct, z):
        sx = self.SxM(ct, st, z)
        sy = self.SyM(ct, st, z)
        sxy = self.sin * sy + self.cos * sx

        o = coef_t()
        o.a = complex(self._mass.a + self._mass.b * sx)
        o.b = complex(self._mass.c + self._mass.d * sxy + o.a)
        o.a = o.a ** 0.5
        o.b = o.b ** 0.5
        return o

    def H(self, st, ct, z, b = 1):
        return z * (self.HTX - b*(self.HTC * ct + self.HTS * st))

    # ---- can only factor Z if m_nu = 0, otherwise the last column has sqrt(Z - m_nu^2)
    def Htilde(self, st, ct, z, b = 1):
        return z * (self.HBX - b*(self.HBC * ct + self.HBS * st))

    def alpha_p(self, st, ct): return self.o * self.spsi * ct + self.lep.b * self.cpsi * st
    def alpha_m(self, st, ct): return self.o * self.cpsi * ct - self.lep.b * self.spsi * st
   
    def P(self, l, z, ct, st, b = 1):
        ap = self.alpha_p(st, ct)
        am = self.alpha_m(st, ct)
        lb = self.lep.b

        o  = (l - z / self.o) * l ** 2
        o += (-b * z ** 2)/(lb * self.o**2) * ( (1 - lb**2) * ap - self.tpsi * am) * l
        o += ( b * z ** 3) / (lb * self.o) * (ap - self.tpsi * am)
        return o 

    def dPtau(self, l, z, ct, st, b = 1):
        o  =  l * (z ** 2 / self.o) * self.alpha_m(st, ct)
        o += - (ct * z ** 3) / (self.o * self.cpsi) 
        return - b * o 

    def dPl0(self, z, ct, st):
        return z * ct / (self.cpsi * self.alpha_m(st, ct))

    def PL0(self, ct, st, b = 1): 
        return self.alpha_p(st, ct) + b * self.lep.b * self.cpsi ** 2 * self.alpha_m(st, ct) ** 2

    @property 
    def R_T(self):
        if self.RT is not None: return self.RT
        px, py, pz = self.lep.px, self.lep.py, self.lep.pz
        phi   = np.arctan2(py, px)
        theta = np.arctan2(np.sqrt(px**2 + py**2), pz)
        R_z   = rotation_z(-phi)
        R_y   = rotation_y(0.5*np.pi - theta)
        
        b_vec = np.array([self.jet.px, self.jet.py, self.jet.pz])
        b_rot = R_y @ (R_z @ b_vec)
        R_x = rotation_x(-np.arctan2(b_rot[2], b_rot[1]))
        self.RT = R_z.T @ R_y.T @ R_x.T
        return self.RT

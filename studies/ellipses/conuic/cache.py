from atomics import *
from mobius import *
import numpy as np
import cmath 

class matrix:
    def __init__(self):
        self.RT = None

    def P(self, z, l, tau):
        z = abs(z)
        a = l**3
        b = - (z/self.o) * l**2
        c = - (z**2) * (l / self.o) * (self._b * self.spsi * cosh(tau) - self.o * self.cpsi * sinh(tau))
        d = - (z**3) * (1 / (self.o * self.cpsi)) * sinh(tau)
        return a + b + c + d

    def dPdt(self, z, l, tau):
        z = abs(z)
        a = - (l * z ** 2) * (1 / self.o) * (self._b * self.spsi * sinh(tau) - self.o * self.cpsi * cosh(tau))
        b = - (z ** 3) * cosh(tau) / (self.o * self.cpsi)
        return a + b

    def dPdtL0(self, z, tau): return - (abs(z) / self.cpsi) * 1.0 / (self._b * self.spsi * tanh(tau) - self.o * self.cpsi)

    def dPL0(self):
        dsc = (1 + 4 * self.lep.b * self.o * self.spsi)**0.5
        f  = 1 + 2 * self.lep.b * self.o * self.spsi * self.cpsi ** 0.5
        ux = (f + self.lep.b * dsc) / (2 * self.lep.b**2 * self.spsi**2 * self.cpsi)
        uv = (f - self.lep.b * dsc) / (2 * self.lep.b**2 * self.spsi**2 * self.cpsi)
        return ux, uv

    def N(self):
        hc = self.Hmatrix(1, self.tau)
        hv = np.linalg.inv(hc)
        return hv.T.dot(np.diag([1, 1, -1])).dot(hv)

    def cache(self):
        self.cos = costheta(self.jet, self.lep)
        self.sin = (1 - self.cos**2) ** 0.5
        self._b = self.lep.b
        self._m = self.lep.mass
        self._e = self.lep.e
        
        self.w   = (self.lep.b/self.jet.b - self.cos)/self.sin
        self.o2  = self.w**2 + 1 - self.lep.b ** 2
        self.o   = self.o2 ** 0.5

        # ------- psi declare ----------#
        self.tpsi = self.w
        self.cpsi = 1.0 / (1 + self.w**2)**0.5
        self.spsi = self.tpsi * self.cpsi
        rt = self.R_T

        # ------- Z2 ----------- #
        self.za = (1 - self.o2) / self.o2
        self.zb = 2 * self.w / self.o2
        self.zc = (self.w**2 - self.o2)/self.o2
        self.zd = 2 * self.lep.e * self.lep.b
        self.ze = self.lep.mass**2 - self.m_nu**2

        # ------- HBAR --------- #
        self.HB1 = nulls(3,3)
        self.HB1[0][0] = 1
        self.HB1[1][0] = self.tpsi 
        self.HB1[2][1] = self.o
        self.HB1 = np.array(self.HB1)
        self.HT1 = rt.dot(self.HB1)
      
        self.HB2 = nulls(3,3)
        self.HB2[0][2] = self._b * self.cpsi
        self.HB2[1][2] = self._b * self.spsi
        self.HB2 = np.array(self.HB2)
        self.HT2 = rt.dot(self.HB2)
     
        self.HB3 = nulls(3,3)
        self.HB3[0][2] =  self.o * self.spsi
        self.HB3[1][2] = -self.o * self.cpsi
        self.HB3 = np.array(self.HB3)
        self.HT3 = rt.dot(self.HB3)

        # ------ poles
        self.pn =  (self.o / self._b) * (1 / self.tpsi) # Asymptote
        self.pm = -(self.o / self._b) * self.tpsi       # Z / Omega solution

        self.mob = mobius(self)

    def test(self, tau):
        self.debug()


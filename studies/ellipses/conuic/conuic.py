from particle import *
from visualize import *
from atomics import *
from cache import *

import numpy as np
import math

class conuic(matrix):

    def __init__(self, lep, bqrk, event_t = None, runtime = None):
        matrix.__init__(self)

        self.l    = 1 # lambda 
        self.z    = 1 # scaling factor Z
        self.tau  = 1 # hyperbolic variable
        self.error = None 
        
        self.lep = lep
        self.jet = bqrk
        self.cache()

        self.truth_pair = []

        self.is_truth = False
        if event_t is None: return self.test(1)
        if self.jet.top_index != self.lep.top_index: return self.test(1)
        if self.lep.top_index not in event_t.truth_pairs: return self.test(1)
        self.truth_pair = event_t.truth_pairs[self.jet.top_index]
        self.is_truth = True
        self.test(1)
            
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

    def Sx(self, z, tau): return (z * self.cpsi / self._b) * (self.o * cosh(tau) - self._b * self.tpsi * sinh(tau)) - (self._m**2)/(self._e * self._b)
    def Sy(self, z, tau): return (z * self.cpsi / self._b) * (self.o * self.tpsi * cosh(tau) - self._b * sinh(tau)) - self.tpsi * self._e / self._b

    def Htilde(self, z, tau):  return (z/self.o) * ( self.HB1 + self.HB2 * cosh(tau)  + self.HB3 * sinh(tau))
    def Hmatrix(self, z, tau): return (z/self.o) * ( self.HT1 + self.HT2 * cosh(tau)  + self.HT3 * sinh(tau))

    def PL0(self, tau):  return self.alpha_P(tau) * cosh(tau) + (self._b * self.cpsi ** 2) * self.alpha_M(tau) ** 2
    def dPdtL0(self, z, tau): return - (z / self.cpsi) / (self._b * self.spsi * tanh(tau) - self.o * self.cpsi)
    def alpha_P(self, tau): return  self.o * self.spsi             + self._b * self.cpsi * tanh(tau)
    def alpha_M(self, tau): return self._b * self.spsi * tanh(tau) - self.o  * self.cpsi

    def mW2(self, z, tau): return self._m ** 2  - 2 * z * self.cpsi * self._e * (self.o * cosh(tau) - self._b * self.tpsi * sinh(tau))
    def mT2(self, z, tau):
        pb = self.jet.e * self.jet.b 
        pm = self._e + self._b
        mt2  = self.jet.mass ** 2 - self._m ** 2 - 2 * ( (pb * self.cos + pm) * self.Sx(z, tau) + pb * self.sin * self.Sy(z, tau) )
        return mt2

    def masses(self, z, tau): return (abs(self.mW2(z, tau)) ** 0.5, abs(self.mT2(z, tau)) ** 0.5)


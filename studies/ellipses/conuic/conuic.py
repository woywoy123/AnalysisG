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

    def Z2(self, Sx, Sy): return self.za * Sx**2 + self.zb * Sy * Sx + self.zc * Sy ** 2 + self.zd * Sx + self.ze
    
    def GetTauZ(self, Sx, Sy):
        f = self.o / self.lep.b
        kx = self.lep.mass ** 2 / (self.lep.e * self.lep.b)
        ky = self.tpsi * self.lep.e / self.lep.b
        a = f * ( (Sy + ky) * self.cpsi - (Sx + kx) * self.spsi )
        b = (Sx + kx) * self.cpsi + (Sy + ky) * self.spsi
        try: t = math.atanh(a / b)
        except ValueError: return None, None
        z = (Sx + kx) / ( ((self.o * self.cpsi)/self.lep.b) * cosh(t) - self.spsi * sinh(t) )
        return z, t


    def eigenvectors(self):
        a1 = self.cpsi * self.lep.b * cosh(self.tau) + self.spsi * self.o * sinh(self.tau)
        mu_s = self.o * (1 + self.tpsi**2) / (self.o - self.lep.b * self.tpsi * tanh(self.tau))
        vstar  = np.array([a1 / (mu_s - 1), mu_s/self.o, 1])
        vpstar = self.RT.dot(vstar) 
        self.theta_star = math.atan2(vpstar[1], vpstar[0])
        return vstar, vpstar


    def Sx(self, z, tau): return (z * self.cpsi / self._b) * (self.o * cosh(tau) - self._b * self.tpsi * sinh(tau)) - (self._m**2)/(self._e * self._b)
    def Sy(self, z, tau): return (z * self.cpsi / self._b) * (self.o * self.tpsi * cosh(tau) + self._b * sinh(tau)) - self.tpsi * self._e / self._b

    def x1(self, z, tau): return self.lep.p + abs(z) * self.cpsi / self.o * (self.lep.b * cosh(tau) + self.o * self.tpsi * sinh(tau))
    def y1(self, z, tau): return            + abs(z) * self.cpsi / self.o * (self.lep.b * self.tpsi * cosh(tau) - self.o * sinh(tau))

    def Htilde(self, z, tau):  return abs(z/self.o) * ( self.HB1 + self.HB2 * cosh(tau) + self.HB3 * sinh(tau) )
    def Hmatrix(self, z, tau): return abs(z/self.o) * ( self.HT1 + self.HT2 * cosh(tau) + self.HT3 * sinh(tau) )

    def mW2(self, z, tau): return complex(- self._m ** 2 - 2 * self.lep.e * self.lep.b * self.Sx(z, tau))
    def mT2(self, z, tau): return self.mW2(z, tau) + self.jet.mass**2 - 2 * self.jet.e * self.jet.b * (self.Sy(z, tau) * self.sin + self.Sx(z, tau) * self.cos)
    def masses(self, z, tau): return abs(self.mW2(z, tau)) ** 0.5, abs(self.mT2(z, tau)) ** 0.5


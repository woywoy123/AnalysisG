from original import NuSol
import numpy as np
import math

from experimental import *
from particle import *
from atomics import * 
from figures import *
from base import *

class algorithm:
    def __init__(self, jet, lep, nu = None):
        self.lep = lep 
        self.jet = jet
        self.nu  = nu
        self.ref = NuSol(self.jet, self.lep, self.nu)
        self.data = data_t(jet, lep)

        self.Z2_pp = Z2_t(self.data, +1, +1)
        self.Z2_pm = Z2_t(self.data, +1, -1)
        self.Z2_mp = Z2_t(self.data, -1, +1)
        self.Z2_mm = Z2_t(self.data, -1, -1)
        self.G2    = G2_t(self.data)

        self.elliptical()

    def elliptical(self):
        def make(tau, phi, s1, s2, eps): 
            tau = tau_Con(self.data, s1)
            phi = cosphi(self.data, tau, s1, s2, eps)
            #phi = Omega(self.data, s1) / (self.data.b_mu * omega(self.data, s1) * np.tanh(tau))
            #if abs(phi) > 1: tau = np.atanh(Omega(self.data, s1) / (self.data.b_mu * omega(self.data, s1) * phi))
#           # phi = Omega(self.data, s1) / (self.data.b_mu * omega(self.data, s1) * np.tanh(tau))
#           # phi = cosphi(self.data, tau, s1, s2, eps)
            def Sigma(data , dt, s1): return np.cos(math.atan(omega(data, s1))) - dt * np.sin(math.atan(omega(data, s1)))
            dt = delta(self.data, +1) 
#            m_nu = self.data.e_mu * (dt * omega(self.data, s1) - (1 - self.data.b_mu**2)) /(Omega(self.data, s1) * Sigma(self.data, dt, s1))
            m_nu = m_nuG(self.data, tau, phi, s1, s2, eps)
            #m_nu = m2_nu(self.data, s1, s2) # phi, s1, s2, eps)
            
            print(self.nu.mass)
            slx = self.G2.Slx(tau, phi, ws=s1, eps=eps, m_nu=m_nu)
            sly = self.G2.Sly(tau, phi, ws=s1, eps=eps, m_nu=m_nu)
            x1, y1 = self.G2.lxly_to_x1y1(slx, sly, sign=s1)
            
            Z2p = self.G2.Z2p(slx, sly, eps=eps, m_nu=m_nu)
            Z2m = self.G2.Z2m(slx, sly, eps=eps, m_nu=m_nu)
            sz = Z2p if s1 > 0 else Z2m
            if Z2p < 0: sz = Z2m
            if Z2m < 0: sz = Z2p
            sz = sz ** 0.5 if sz > 0 else abs(sz)**0.5 
            
            sz = m_nu * np.sinh(tau) * np.sin(phi)
            O = Omega(self.data, s1)
            w = omega(self.data, s1)
            rx = self.G2.ev
            x = np.array([
                [sz / O,     0, x1*0 - self.data.p_mu*0], 
                [w * sz / O, 0, y1*0                 ],
                [0,          sz, 0                 ]
            ]) 
#            d = x.T.dot((np.diag(self.G2.ei)))
#            print(self.G2.ei.T)
#            x = d.T.dot(np.diag(self.G2.ei)).dot(d)
            return x, x.dot([np.cos(phi), np.sin(phi), 1])# / np.sqrt(3) #np.array([x1, y1, sz])

        def lpppp(tau, phi): return make(tau, phi, +1, +1, +1)
        def lpppm(tau, phi): return make(tau, phi, +1, +1, -1)
        def lppmp(tau, phi): return make(tau, phi, +1, -1, +1)
        def lppmm(tau, phi): return make(tau, phi, +1, -1, -1)

        def lpmpp(tau, phi): return make(tau, phi, -1, +1, +1)
        def lpmpm(tau, phi): return make(tau, phi, -1, +1, -1)
        def lpmmp(tau, phi): return make(tau, phi, -1, -1, +1)
        def lpmmm(tau, phi): return make(tau, phi, -1, -1, -1)

        a = np.pi/2
        data = self.ref.solution
        pkt = packet(data["H_T"], data["neutrino"])

        h, s = lpppp(10, a)
        pkt.add_ellipse(h, "H++++", s)
        h, s = lpppm(10, a)
        pkt.add_ellipse(h, "H+++-", s)
        h, s = lppmp(10, a)
        pkt.add_ellipse(h, "H++-+", s)
        h, s = lppmm(10, a)
        pkt.add_ellipse(h, "H++--", s)

        h, s = lpmpp(10, a)
        pkt.add_ellipse(h, "H+-++", s)
        h, s = lpmpm(10, a)
        pkt.add_ellipse(h, "H+-+-", s)
        h, s = lpmmp(10, a)
        pkt.add_ellipse(h, "H+--+", s)
        h, s = lpmmm(10, a)
        pkt.add_ellipse(h, "H+---", s)

        pkt.compile2D_Proj(None, True)

    @property 
    def R_T(self):
        px, py, pz = self.lep.px, self.lep.py, self.lep.pz
        phi   = np.arctan2(py, px)
        theta = np.arctan2(np.sqrt(px**2 + py**2), pz)
        R_z   = angular_t(-phi).Rz
        R_y   = angular_t(0.5*np.pi - theta).Ry
        
        b_vec = np.array([self.jet.px, self.jet.py, self.jet.pz])
        b_rot = R_y @ (R_z @ b_vec)
        R_x = angular_t(-np.arctan2(b_rot[2], b_rot[1])).Rx
        self.RT = R_z.T @ R_y.T @ R_x.T
        return self.RT




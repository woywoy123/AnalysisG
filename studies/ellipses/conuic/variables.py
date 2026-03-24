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
        mw = self.ref.mW2 ** 0.5
        mt = self.ref.mT2 ** 0.5
        mn = self.ref.mN2 ** 0.5
        def make(tau, phi, s1, s2, eps): 
            #            tau = tau_H(self.data, s1, s2)
            m_nu = m_nuG(self.data, tau, phi, s1=s1, s2=s2, eps=eps)
#            print(dLdtau(self.data, m_nu, tau, phi, s1, s2, eps)) 
            slx = self.G2.Slx(tau, phi, ws=s1, eps=eps, m_nu=m_nu)
            sly = self.G2.Sly(tau, phi, ws=s1, eps=eps, m_nu=m_nu)
            x1, y1 = self.G2.lxly_to_x1y1(slx, sly, sign=s1)
            if s1 > 0: Z2 = self.G2.Z2p(slx, sly, eps=eps, m_nu=m_nu)
            else:      Z2 = self.G2.Z2m(slx, sly, eps=eps, m_nu=m_nu)

            sx, sy = self.G2.lxly_to_SxSy(slx, sly)
            _mw, _mt, _mn = mW(self.data, sx, m_nu), mT(self.data, sx, sy, m_nu), m_nu
#            if (_mw + _mt + _mn).imag != 0: return 
            return (_mw.real - mw) ** 2 + (_mt.real - mt) ** 2 + (_mn - mn) ** 2 


            #sz = m_nu * np.sinh(tau) * np.sin(phi)
            sz = Z2 ** 0.5 if Z2 > 0 else complex(Z2)**0.5 
            O = Omega(self.data, s1)
            w = omega(self.data, s1)
            x = np.array([
                [sz / O,     0, x1 - self.data.p_mu], 
                [w * sz / O, 0, y1                 ],
                [0,          sz, 0                 ]
            ])
            x = x.dot(x.T)
            return x


        def lpppp(tau, phi): return make(tau, phi, +1, +1, +1)
        def lpppm(tau, phi): return make(tau, phi, +1, +1, -1)
        def lppmp(tau, phi): return make(tau, phi, +1, -1, +1)
        def lppmm(tau, phi): return make(tau, phi, +1, -1, -1)

        def lpmpp(tau, phi): return make(tau, phi, -1, +1, +1)
        def lpmpm(tau, phi): return make(tau, phi, -1, +1, -1)
        def lpmmp(tau, phi): return make(tau, phi, -1, -1, +1)
        def lpmmm(tau, phi): return make(tau, phi, -1, -1, -1)


        mnx = -1
        p = []
        for i in range(1000):
            a = (i / 1000) * np.pi 
            for j in range(100000):
                k = -10 + j * 0.0001
                x = lpppp(k, a)
                if x > mnx and mnx > 0: continue
                p = [a, k, x]
                mnx = x
                print(mnx, p)

            continue

            data = self.ref.solution
            pkt = packet(data["H_T"], data["neutrino"])
            pkt.add_ellipse(lpppp(10, a), "H++++")
            pkt.add_ellipse(lpppm(10, a), "H+++-")
            pkt.add_ellipse(lppmp(10, a), "H++-+")
            pkt.add_ellipse(lppmm(10, a), "H++--")

            pkt.add_ellipse(lpmpp(10, a), "H+-++")
            pkt.add_ellipse(lpmpm(10, a), "H+-+-")
            pkt.add_ellipse(lpmmp(10, a), "H+--+")
            pkt.add_ellipse(lpmmm(10, a), "H+---")

            pkt.compile2D_Proj(None, True)
        exit()

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




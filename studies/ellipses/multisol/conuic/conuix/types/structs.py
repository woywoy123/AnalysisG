from conuix.types.base import basics_t
from conuix.base.constants import *
from conuix.base.atomics import *
from conuix.base.utils import *
import numpy as np
import math

class lambda_t(basics_t):

    def __init__(self, data):
        basics_t.__init__(self, data.bq, data.lp)

        self.phi  = data.delta_phi
        self.GG   = self.Gp * self.Gm
        self.gm   = -self.dp * self.dm
        self.ch   = np.cosh(self.phi)
        self.sh   = np.sinh(self.phi)

        self.k_m = np.atan(self.wm)
        self.k_p = np.atan(self.wp)

        self.lxp = self.Lx(data.Sx0, data.Sy0(+1))
        self.lxm = self.Lx(data.Sx0, data.Sy0(-1))

        self.lyp = self.Ly(data.Sx0, data.Sy0(+1))
        self.lym = self.Ly(data.Sx0, data.Sy0(-1))

        self.Sz = 2 * (self.wp - self.wm) / (self.Op * self.Om)

        self.N = np.sqrt( -(self.sh * self.ch) / self.GG ) * 1.0 / np.cosh(2 * self.phi)
        self.T = np.array([
            [self.N * self.ch,  self.N * self.sh * self.dp * self.dm,       0], 
            [self.N * self.sh, -self.N * self.ch * self.dp * self.dm,       0], 
            [0              ,                                      0, self.Sz]
        ])
        self.htilde = data.htilde

    def Lx(self, sx, sy):
        c = np.sqrt( - self.GG / (self.sh * self.ch) )
        x = self.ch * sx
        y = self.sh * sy
        return c * (x - np.sqrt(self.gm) * y)
        
    def Ly(self, sx, sy):
        c = np.sqrt( - self.GG / (self.sh * self.ch) )
        x = self.sh * sx
        y = self.ch * sy
        return c * (x + np.sqrt(self.gm) * y)

    def Sx(self, lx, ly):
        c = np.sqrt( -(self.sh * self.ch) / self.GG )
        x = self.ch * lx / np.cosh(2 * self.phi)
        y = self.sh * ly / np.cosh(2 * self.phi)
        return c * (x + y)

    def Sy(self, lx, ly):
        c = np.sqrt( -(self.sh * self.ch) / self.GG )
        x = -self.sh * lx / (np.sqrt(self.gm) * np.cosh(2 * self.phi))
        y =  self.ch * ly / (np.sqrt(self.gm) * np.cosh(2 * self.phi))
        return c * (x + y)

    def Z2P(self, lx, ly, m_nu, lz = 0, lSz = 0):
        K = -(self.sh * self.ch) / (self.GG * (self.Op * np.cosh(2 * self.phi))**2) 
        k1 =  1 - (self.Op * self.ch)**2 - 2 * self.wp / np.sqrt(self.gm) * self.ch * self.sh
        k3 = -1 - (self.Op * self.sh)**2 + 2 * self.wp / np.sqrt(self.gm) * self.ch * self.sh
        k2 = (1 + self.b_mu ** 2 - self.wp ** 2) * self.ch * self.sh + self.wp / np.sqrt(self.gm)

        lin = np.sqrt(-(self.sh * self.ch) / self.GG) / np.cosh(2 * self.phi)
        k4 =  2 * self.p_mu * self.ch * lin
        k5 =  2 * self.p_mu * self.sh * lin
        Q  = K * ( k1 * lx ** 2 + 2 * k2 * lx * ly + k3 * ly ** 2 ) + k4 * lx + k5 * ly
        Q += self.m_mu ** 2 - m_nu ** 2 - (lz - (1 / self.Sz) * lSz) ** 2
        return Q

    def Z2M(self, lx, ly, m_nu, lz = 0, lSz = 0):
        K = -(self.sh * self.ch) / (self.GG * (self.Om * np.cosh(2 * self.phi))**2) 
        k1 =  1 - (self.Om * self.ch)**2 - 2 * self.wm / np.sqrt(self.gm) * self.ch * self.sh
        k3 = -1 - (self.Om * self.sh)**2 + 2 * self.wm / np.sqrt(self.gm) * self.ch * self.sh
        k2 = (1 + self.b_mu ** 2 - self.wm ** 2) * self.ch * self.sh + self.wm / np.sqrt(self.gm)

        lin = np.sqrt(-(self.sh * self.ch) / self.GG) / np.cosh(2 * self.phi)
        k4 =  2 * self.p_mu * self.ch * lin
        k5 =  2 * self.p_mu * self.sh * lin
        Q  = K * ( k1 * lx ** 2 + 2 * k2 * lx * ly + k3 * ly ** 2 ) + k4 * lx + k5 * ly
        Q += self.m_mu ** 2 - m_nu ** 2 - (lz - self.Sz * lSz) ** 2
        return Q

    def Htilde(self, lx, ly, lz, s1, lSz, m_nu):
        lx, ly = (lx - self.lxp, ly - self.lyp) if s1 > 0 else (lx - self.lxm, ly - self.lym)
        z2 = self.Z2P(lx, ly, m_nu, lz, lSz) if s1 > 0 else self.Z2M(lx, ly, m_nu, lz, lSz)
        sx, sy = self.Sx(lx, ly), self.Sy(lx, ly)
        return self.htilde(sx, sy, abs(z2)**0.5, s1)



class linear_t(basics_t):

    def __init__(self, data):
        basics_t.__init__(self, data.bq, data.lp)
        self.rho = data.Om / data.Op
        self.frx = self.m_mu * (data.Om + self.Op) / (2 * np.sqrt(self.Om * self.Op))  
     
        self.mP = mx_Pnu(data, self._branch(+1), +1)
        self.mM = mx_Pnu(data, self._branch(-1), -1)

    def _branch(self, br): 
        try: m = self.mP if br > 0 else self.mM
        except AttributeError: m = None
        w, O = (self.wp, self.Op) if br > 0 else (self.wm, self.Om)
        return {"w" : w, "O" : O, "M" : m, "N" : O ** 2 * (self.dp - self.dm)}

    def alpha(self, m_nu, br, s):
        a = 1  + s * self.rho ** br
        O = self.Op if br > 0 else self.Om
        b = (self.wp - self.wm)/(self.b_mu * O)
        return 0.5 * np.sqrt( (a ** 2 + ( (self.m_mu / m_nu) ** 2 - 1 ) * b ** 2) )

    def lF(self, m_nu, s, tnh = +1): 
        u = (self.alpha(m_nu, -1, tnh) / self.alpha(m_nu, +1, tnh))
        return (self.rho ** tnh) * ( (1 + u) / (1 - u) ) ** s

    def aM(self, eta):     return np.tanh(eta / 2) * self.rho
    def aP(self, eta):     return np.cosh(eta / 2) * self.rho
    def eta(self, am, ap): return 2 * np.atanh( (am / ap) ** ( am > ap ) )

    def m_nu(self, eta):   return self.frx * 1.0 / np.cosh(eta / 2.0)
    def t_chi(self, nv, br):
        _br = self._branch(br)
        V  = np.linalg.inv(_br["M"][:,:4]).dot( (nv - _br["M"][:,4]).reshape((4, 1)) ).flatten()
        return np.atan2(V[3], V[2])

    def t_tau(self, nv, br):
        _br = self._branch(br)
        V  = np.linalg.inv(_br["M"][:,:4]).dot( (nv - _br["M"][:,4]).reshape((4, 1)) ).flatten()
        m_nu = nv[3]**2 - sum([nv[i] ** 2 for i in range(3)])
        return np.asinh( ((V[2] ** 2 + nv[3] ** 2) / m_nu) ** 0.5 )

    def Sx(self, tau, eta, chi, br, sg):
        o, k = (self.Op, np.atan(self.wp)) if br > 0 else (self.Om, np.atan(self.wm))
        u  = sg * o * np.cos(k) * np.cosh(tau) - self.b_mu * np.sin(k) * np.cos(chi) * np.sinh(tau) 
        return (self.m_nu(eta) / self.b_mu) * u + self.Sx0

    def Sy(self, tau, eta, chi, br, sg):
        o, k = (self.Op, np.atan(self.wp)) if br > 0 else (self.Om, np.atan(self.wm))
        u  = sg * o * np.sin(k) * np.cosh(tau) + self.b_mu * np.cos(k) * np.cos(chi) * np.sinh(tau) 
        return (self.m_nu(eta) / self.b_mu) * u + ( self.Sy0p if br > 0 else self.Sy0m )
    
    def Sz(self, tau, eta): return self.m_nu(eta) * np.sinh(tau)

    def P_nu(self, tau, eta, chi, br, sg):
        mx = self.mP if br > 0 else self.mM
        sx, sy, sz = self.Sx(tau, eta, chi, br, sg), self.Sy(tau, eta, chi, br, sg), self.Sz(tau, eta)
        v = np.array([sx - self.dp * sy, sx - self.dm * sy, sz * np.cos(chi), sz * np.sin(chi), 1])
        return mx.dot(v)


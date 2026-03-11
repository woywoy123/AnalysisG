from constants import *
from atomics import *
from classes import *
from matrix import *
import math

class Z2_t:
    def __init__(self, data, sign, eps):
        self.w     = omega(data, sign)
        self.o     = Omega(data, sign)
        self.b_mu  = data.b_mu
        self.p_mu  = data.p_mu
        self.kappa = angular_t(math.atan(self.w))
        self.dt    = delta(data, eps)

        self.xkp   = sign
        self.eps   = eps 
        self.data  = data
        
        self.ls  = Z2_coeffs(data, sign)
        self.Sx0 = Sx0(data)
        self.Sy0 = Sy0(data, sign) 

    def Z2(self, sx, sy, m_nu = 0):
        Sp = [sx**2, sx*sy, sy**2, sx, 1]
        return sum([i * j for i, j in zip(self.ls, Sp)]) - m_nu ** 2

    def Sx(self, tau, phi, m_nu = 1, cosphi = False):
        hx, kp = hyper_t(tau), self.kappa
        cphi = cosphi if phi is not False else np.cos(phi)
        sx = self.eps * self.o * kp.cos * hx.cosh - self.b_mu * cphi * kp.sin * hx.sinh
        return (m_nu / self.b_mu) * sx + self.Sx0

    def Sy(self, tau, phi, m_nu = 1, cosphi = False):
        hx, kp = hyper_t(tau), self.kappa
        cphi = cosphi if phi is not False else np.cos(phi)
        sy = self.eps * self.o * kp.sin * hx.cosh + self.b_mu * cphi * kp.cos * hx.sinh
        return (m_nu / self.b_mu) * sy + self.Sy0

    def Sz(self, tau, phi, m_nu = 1):
        return self.eps * m_nu * np.sinh(tau) * np.sin(phi)

    def x1(self, tau, phi, m_nu = 1, cosphi = False):
        hx, kp = hyper_t(tau), self.kappa
        kx = self.eps * self.b_mu * kp.cos * hx.cosh - self.o * kp.sin * np.cos(phi) * hx.sinh
        return self.p_mu - (m_nu / self.o)*kx

    def y1(self, tau, phi, m_nu = 1, cosphi = False):
        hx, kp = hyper_t(tau), self.kappa
        ky =  self.o * kp.cos * np.cos(phi) * hx.sinh - self.eps * kp.sin * self.b_mu * hx.cosh
        return (m_nu / self.o)*ky

    def S_M(self, Sx, Sy):
        return H_matrix(self.data, self.xkp).dot(np.array([Sx.Sy]))


class G2_t:
    def __init__(self, data):
        self.w   = branch_t(omega(data,+1), omega(data,-1))
        self.o   = branch_t(Omega(data,+1), Omega(data,-1))
        self.G   = branch_t(Gamma(data,+1), Gamma(data,-1))
        self.dt  = branch_t(delta(data,+1), delta(data,-1))
        self.psi = branch_t(angular_t(math.atan(self.dt.p)), angular_t(math.atan(self.dt.m)))
        
        f = Gamma(data, +1) * Gamma(data, -1) / (self.psi.m.cos * self.psi.p.cos)
        self.eig = branch_t(
                - f * np.cos( (self.psi.p.alpha - self.psi.m.alpha) / 2 )**2,
                  f * np.sin( (self.psi.p.alpha - self.psi.m.alpha) / 2 )**2
        )
        self.data = data

        self.MX = np.array([
            [Mxy(data, +1, +1), Mxy(data, +1, -1)], 
            [Mxy(data, -1, +1), Mxy(data, -1, -1)]
        ])
        self.ei, self.ev = np.linalg.eig(self.MX)

    def dG2(self, Sx, Sy): 
        return - self.G.m * self.G.p * (Sx - self.dt.p * Sy) * (Sx - self.dt.m * Sy)


    def _Line(self, tau, eps, s1, s2, phi=False):
        z2 = Z2_t(self.data, s1, s2)
        cphi = cosphi(self.data, eps, np.atanh(eta_test(self.data, eps, s1, s2)), s1, s2) if phi is None else phi
        m_nu = m_nuG(self.data, eps, np.atanh(eta_test(self.data, eps, s1, s2)), s1, s2)

        # 1. Generate the quadric Sx, Sy coordinates
        sx = z2.Sx(tau, cphi, m_nu, phi is not None)
        sy = z2.Sy(tau, cphi, m_nu, phi is not None)

        return H_tilde(sx, sy, complex(z2.Z2(sx, sy, m_nu)) ** 0.5, self.data, s1)



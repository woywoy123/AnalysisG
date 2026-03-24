from constants import *
from atomics import *
from classes import *
from matrix import *
import math

def eta_test(data, eps, s1, s2):
    dt, w, O = delta(data, s2), omega(data, s1), Omega(data, s2)
    v = eps * (data.b_mu / O) * (w + dt) / ( 1 - dt * w)
    return v if abs(v) < 1 else 1 / v

def m_nuG(data, tau, phi, s1, s2, eps):
    dt, kappa, O = delta(data, s2), np.atan(omega(data, s1)), Omega(data, s1)
    skp, ckp, tkp = np.sin(kappa), np.cos(kappa), np.tan(kappa)
    a = dt * tkp * data.e_mu ** 2 - data.m_mu ** 2 
    b = data.p_mu * (skp + dt * ckp) * np.sinh(tau) * np.cos(phi) - eps * data.e_mu * O * (ckp - dt * skp) * np.cosh(tau)
    return a / b

def Mxy(data, s1, s2):
    w = omega(data, s1)
    dt = delta(data, s2)
    b2 = data.b_mu**2
    gamma = 1.0 - b2
    return (gamma - dt * w) / (dt * (w**2 - b2) - w)


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
        self.lsp = Z2_t(self.data, +1, +1).ls
        self.lsm = Z2_t(self.data, -1, -1).ls

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


    def Z2p(self, lx, ly, eps, m_nu):
        dtp, dtm = self.dt.p, self.dt.m
        A, B, C = self.lsp[0], self.lsp[1], self.lsp[2]
        a = (A * dtm ** 2 + B * dtm + C)/(dtp - dtm) ** 2 
        b = (-2 * A * dtp * dtm - B * (dtp + dtm) - 2 * C) / (dtp - dtm)**2
        c = (A * dtp ** 2 + B * dtp + C) / (dtp - dtm) ** 2 
        d = -2 * self.data.p_mu * dtm / (dtp - dtm) 
        e = +2 * self.data.p_mu * dtp / (dtp - dtm) 
        f = self.data.m_mu ** 2 - m_nu ** 2 
        return a * lx ** 2 + b * lx * ly + c * ly ** 2 + d * lx + e * ly + f 

    def Z2m(self, lx, ly, eps, m_nu):
        dtp, dtm = self.dt.p, self.dt.m
        A, B, C = self.lsm[0], self.lsm[1], self.lsm[2]
        a = (A * dtm ** 2 + B * dtm + C)/(dtp - dtm) ** 2 
        b = (-2 * A * dtp * dtm - B * (dtp + dtm) - 2 * C) / (dtp - dtm)**2
        c = (A * dtp ** 2 + B * dtp + C) / (dtp - dtm) ** 2 
        d = -2 * self.data.p_mu * dtm / (dtp - dtm) 
        e = +2 * self.data.p_mu * dtp / (dtp - dtm) 
        f = self.data.m_mu ** 2 - m_nu ** 2 
        return a * lx ** 2 + b * lx * ly + c * ly ** 2 + d * lx + e * ly + f 

    def lxly_to_SxSy(self, lx, ly):
        sx = - self.dt.m * lx + self.dt.p * ly
        sy = - lx + ly 
        return sx / (self.dt.p - self.dt.m), sy / (self.dt.p - self.dt.m)

    def lxly_to_x1y1(self, lx, ly, sign):
        dtp, dtm = self.dt.p, self.dt.m
        O, w = Omega(self.data, sign), omega(self.data, sign)
        lxx1 = (dtm * (1  - O ** 2) + w) / (O ** 2 * (dtp - dtm))
        lxy1 = (dtp * (O ** 2 - 1) -  w) / (O ** 2 * (dtp - dtm))

        lyx1 = (- O ** 2 + w * dtm + w ** 2)/(O**2 * (dtp - dtm)) 
        lyy1 = (O** 2  -  w * dtp - w ** 2 )/(O**2 * (dtp - dtm))
        return lxx1 * lx + lxy1 * ly, lyx1 * lx + lyy1 * ly

    def Slx(self, tau, phi, ws, eps, m_nu):
        e_mu, p_mu, m_mu, b_mu = self.data.e_mu, self.data.p_mu, self.data.m_mu, self.data.b_mu
        w, O = omega(self.data, ws), Omega(self.data, ws)
        dtp = self.dt.p
        c = (dtp * w * e_mu ** 2 - m_mu**2) / p_mu
        f = m_nu / (1 + w ** 2 ) ** 0.5 * ((eps * O / b_mu ) * (1 - dtp * w) * np.cosh(tau) - (w + dtp) * np.sinh(tau) * np.cos(phi))
        return c + f

    def Sly(self, tau, phi, ws, eps, m_nu):
        e_mu, p_mu, m_mu, b_mu = self.data.e_mu, self.data.p_mu, self.data.m_mu, self.data.b_mu
        w, O = omega(self.data, ws), Omega(self.data, ws)
        dtm = self.dt.m
        c = (dtm * w * e_mu ** 2 - m_mu**2) / p_mu
        f = m_nu / (1 + w ** 2 ) ** 0.5 * ((eps * O / b_mu ) * (1 - dtm * w) * np.cosh(tau) - (w + dtm) * np.sinh(tau) * np.cos(phi))
        return c + f


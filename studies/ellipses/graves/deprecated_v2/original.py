from particle import *
from atomics import *
import numpy as np
import math

def R(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    R = c * np.eye(3)
    for i in [-1, 0, 1]: R[(axis - i) % 3, (axis + i) % 3] = i * s + (1 - i * i)
    return R


class NuSol(object):
    def __init__(self, b, mu, mW2, mT2, mN2 = 0, truth = None):
        self._b = b
        self._mu = mu
        self.mW2 = mW2*mW2
        self.mN2 = mN2*mN2
        self.mT2 = mT2*mT2
        self._sx = None
        self._sy = None
        self.use_minus = False
        self.truth = truth

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
        P = self.mu.b * self.mu.e
        beta = self.mu.b
        return (self.x0 * beta - P * (1 - beta**2))/(beta**2)

    @property
    def Sy(self):
        beta = self.b.b
        return ((self.x0p / beta) - self.c * self.Sx) / self.s

    @property
    def w(self):
        beta_m, beta_b = self.mu.b, self.b.b
        return (beta_m/beta_b - self.c)/self.s if not self.use_minus else self.wm

    @property
    def wm(self):
        beta_m, beta_b = self.mu.b, self.b.b
        return -(beta_m/beta_b + self.c)/self.s

    @property
    def Om2(self): return self.w**2 + 1 - self.mu.b**2

    @property
    def eps2(self):
        return (self.mW2 - self.mN2) * (1 - self.mu.b**2)

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
    def Z(self): return math.sqrt(max(0, self.Z2))

    @property
    def H_tilde(self):
        return np.array([
            [self.Z/math.sqrt(self.Om2)           , 0   , self.x1 - self.mu.e * self.mu.b],
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
        x = self.H
        f = np.array([self.truth.px, self.truth.py, self.truth.pz])
        k = f

        t = np.linspace(0, 2 * np.pi, 100000)
        f = f * np.concatenate((np.ones_like(t), np.ones_like(t), np.ones_like(t))).reshape(-1, 3)
        df = ((f - (x.dot(np.array([np.cos(t), np.sin(t), np.ones_like(t)])).T) )**2).sum(-1)
        idx = np.argmin(df)

        return {
                "chi2" : df[idx], 
                "angle" : t[idx], 
                "truth" : k, 
                "htilde": self.H_tilde.dot([np.cos(t[idx]), np.sin(t[idx]), np.ones_like(t[idx])]), 
                "reco": x.dot(np.array([np.cos(t[idx]), np.sin(t[idx]), np.ones_like(t[idx])]))
        }



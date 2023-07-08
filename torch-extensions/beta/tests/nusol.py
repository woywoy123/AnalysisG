import vector
import math
import numpy as np
mW = 80.385*1000
mT = 172.0*1000
mN = 0

def costheta(v1, v2):
    v1_sq = v1.x**2 + v1.y**2 + v1.z**2
    if v1_sq == 0: return 0

    v2_sq = v2.x**2 + v2.y**2 + v2.z**2
    if v2_sq == 0: return 0

    v1v2 = v1.x*v2.x + v1.y*v2.y + v1.z*v2.z
    return v1v2/math.sqrt(v1_sq * v2_sq)

def UnitCircle(): return np.diag([1, 1, -1])

def R(axis, angle):
    """Rotation matrix about x(0),y(1), or z(2) axis"""
    c, s = math.cos(angle), math.sin(angle)
    R = c * np.eye(3)
    for i in [-1, 0, 1]:
        R[(axis - i) % 3, (axis + i) % 3] = i * s + (1 - i * i)
    return R


class NuSol(object):
    def __init__(self, b, mu, ev = None, mW2 = mW**2, mT2 = mT**2, mN2 = mN**2):
        self._b = b
        self._mu = mu
        self.mW2 = mW2
        self.mT2 = mT2
        self.mN2 = mN2
        self.METx = ev.px
        self.METy = ev.py

    @property
    def b(self): return self._b

    @property
    def mu(self): return self._mu

    @property
    def c(self): return costheta(self._b, self._mu)

    @property
    def s(self): return math.sqrt(1 - self.c**2)

    @property
    def x0p(self): 
        m2 = self.b.tau2
        return -(self.mT2 - self.mW2 - m2)/(2*self.b.e)

    @property
    def x0(self): 
        m2 = self.mu.tau2
        return -(self.mW2 - m2 - self.mN2)/(2*self.mu.e)

    @property
    def Sx(self):
        P = self.mu.mag
        beta = self.mu.beta
        return (self.x0 * beta - P * (1 - beta**2))/beta**2
 
    @property
    def Sy(self):
        beta = self.b.beta
        return ((self.x0p / beta) - self.c * self.Sx) / self.s

    @property
    def w(self):
        beta_m, beta_b = self.mu.beta, self.b.beta
        return (beta_m/beta_b - self.c)/self.s

    @property
    def Om2(self): return self.w**2 + 1 - self.mu.beta**2

    @property
    def eps2(self):
        return (self.mW2 - self.mN2) * (1 - self.mu.beta**2)

    @property
    def x1(self):
        return self.Sx - ( self.Sx + self.w * self.Sy)/self.Om2

    @property
    def y1(self):
        return self.Sy - ( self.Sx + self.w * self.Sy)*self.w/self.Om2

    @property
    def Z2(self):
        p1 = (self.x1**2)* self.Om2 
        p2 = - (self.Sy - self.w * self.Sx)**2
        p3 = - (self.mW2 - self.x0**2 - self.eps2)
        return  p1 + p2 + p3

    @property
    def Z(self): return math.sqrt(max(0, self.Z2))

    @property 
    def BaseMatrix(self):
        return np.array([
            [self.Z/math.sqrt(self.Om2)           , 0   , self.x1 - self.mu.mag], 
            [self.w * self.Z / math.sqrt(self.Om2), 0   , self.y1              ], 
            [0,                                   self.Z, 0                    ]])
    
    @property 
    def X(self):
        S2 = np.vstack([np.vstack([np.linalg.inv([[100, 9], [50, 100]]), [0, 0]]).T, [0, 0, 0]])
        V0 = np.outer([self.METx, self.METy, 0], [0, 0, 1])
        dNu = V0 - self.H
        return np.dot(dNu.T, S2).dot(dNu)

    @property
    def R_T(self):
        b_xyz = self.b.x, self.b.y, self.b.z
        R_z = R(2, -self.mu.phi)
        R_y = R(1, 0.5 * math.pi - self.mu.theta)
        R_x = next(R(0, -math.atan2(z, y)) for x, y, z in (R_y.dot(R_z.dot(b_xyz)),))
        return R_z.T.dot(R_y.T.dot(R_x.T))

    @property
    def H(self):
        return self.R_T.dot(self.BaseMatrix)

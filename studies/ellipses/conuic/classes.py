from atomics   import *
from constants import *

import numpy as np

class hyper_t:
    def __init__(self, tau):
        self.cosh = np.cosh(tau)
        self.sinh = np.sinh(tau)
        self.tanh = np.tanh(tau)

    @property
    def Rz(self):
        return np.array([
            [self.cosh, self.sinh, 0], 
            [self.sinh, self.cosh, 0], 
            [0        ,         0, 1]
        ], np.longdouble)
    
    @property
    def Ry(self):
        return np.array([
            [self.cosh,  0, self.sinh], 
            [0        ,  1,         0], 
            [self.sinh,  0, self.cosh]
        ], np.longdouble)
 
    @property
    def Rx(self):
        return np.array([
            [1,         0,         0], 
            [0, self.cosh, self.sinh], 
            [0, self.sinh, self.cosh]
        ], np.longdouble)


class branch_t:
    def __init__(self, v1, v2):
        self.p = v1
        self.m = v2

class data_t:
    def __init__(self, jet, lep):
        self.theta = angular_t(np.acos(costheta(jet, lep)))

        self.m_mu = lep.mass
        self.b_mu = lep.b
        self.p_mu = lep.p
        self.e_mu = lep.e
        
        self.m_b = jet.mass
        self.b_b = jet.b
        self.p_b = jet.p

class angular_t:
    def __init__(self, alpha):
        self.cos = np.cos(alpha)
        self.sin = np.sin(alpha)
        self.tan = np.tan(alpha)
        self.alpha = alpha
   
    @property
    def Rz(self):
        return np.array([
            [self.cos, -self.sin, 0], 
            [self.sin,  self.cos, 0], 
            [0       ,         0, 1]
        ], np.longdouble)
    
    @property
    def Ry(self):
        return np.array([
            [self.cos ,  0, self.sin], 
            [0        ,  1,        0], 
            [-self.sin,  0, self.cos]
        ], np.longdouble)
 
    @property
    def Rx(self):
        return np.array([
            [1,        0,         0], 
            [0, self.cos, -self.sin], 
            [0, self.sin,  self.cos]
        ], np.longdouble)


class linear_t:
    
    def __init__(self, p1, p2):
        self.d  = p1 - p2 
        self.r0 = p2
    
    def __call__(self, tau): return self.d * tau + self.r0

    def rotx(self, tau, pts): 
        return hyper_t(tau).Rx.dot(pts)

    def rot(self, eta, tau, pts): 
        return hyper_t(eta).Rx.dot(hyper_t(tau).Rx.dot(self(pts)))




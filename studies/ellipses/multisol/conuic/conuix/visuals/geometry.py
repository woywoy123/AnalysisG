from conuix.visuals.visual import cfx_t, cfg_t
from conuix.visuals.atomics import *
import numpy as np

class Point(cfx_t):
    def __init__(self, name, color, scale, ptx):
        cfx_t.__init__(self, name, color, scale)
        self.Q = np.diag([1.0, 1.0, 1.0, -1.0])
        self.pts = 150
        self.px  = ptx[0]
        self.py  = ptx[1]
        self.pz  = ptx[2]
        self.e_  = ptx[3]
        self.m_  = (self.e_ ** 2 - self.px ** 2 - self.py ** 2 - self.pz ** 2)**0.5

    def fx(self, x, y, z): 
        v_ = np.array([self.px, self.py, self.pz,    self.m_ - self.e_ ], dtype = np.float64)
        v  = np.array([self.px, self.py, self.pz, - (self.m_ + self.e_)], dtype = np.float64)
        return FT(x, y, z, v_, v, self.Q)

class ParticleQ(cfx_t):
    def __init__(self, name, color, scale, Q, afx):
        cfx_t.__init__(self, name, color, scale)
        self.pts = 150
        self.Q = Q # 4 x 4 
        self.afx = afx 

    def fx(self, x, y, z): 
        return ShiftPQl(x, y, z, self.Q, self.afx)

class ParticleE(cfx_t):
    def __init__(self, name, color, scale, Q):
        cfx_t.__init__(self, name, color, scale)
        self.pts = 200
        self.Q = Q # 4 x 4 

    def fx(self, x, y, z): 
        return FX(x, y, z, self.Q)

class Ellipse(cfx_t):
    
    def __init__(self, name, color, scale, Q):
        cfx_t.__init__(self, name, color, scale)
        self.pts = 150
        self.Q = Q # 4 x 4 

class TwoSheet(cfx_t):
    def __init__(self, name, color, scale, Q):
        cfx_t.__init__(self, name, color, scale)
        self.pts = 150
        self.Q = Q # 4 x 4 
    def fx(self, x, y, z): return FX(x, y, z, self.Q) 

class TwoSheetT(cfx_t):
    def __init__(self, name, color, scale, Q, T):
        cfx_t.__init__(self, name, color, scale)
        self.pts = 150
        self.Q = Q 
        self.T = T

    def fx(self, x, y, z): 
        return FX(x - self.T[0], y - self.T[1], z, self.Q) - self.T[2] 

# ------- Quadric post translation ------- #
class TwoSheetS0(cfx_t):
    def __init__(self, name, color, scale, Q, brch, s):
        cfx_t.__init__(self, name, color, scale)
        self.pts = 150
        self.Q = Q # 4 x 4 
        self.Sx0 = brch.Sx0
        self.Sy0 = brch.Sy0(s)

    # ------- Shift to center ----- #
    def sxT(self, x, y, z): return x
    def syT(self, x, y, z): return y
    def fx(self, x, y, z): return FX(x + self.Sx0, y + self.Sy0, z, self.Q)


# ------- Quadric post translation + rotation ------- #
class TwoSheetS0R(cfx_t):
    def __init__(self, name, color, scale, Q, brch, s):
        cfx_t.__init__(self, name, color, scale)
        self.Q = Q
        self.pts = 120
        self.Sx0 = brch.Sx0 
        self.Sy0 = brch.Sy0(s)
        self.Sz0 = 0

        kappa = np.atan(brch.wp if s > 0 else brch.wm)
        self.Rn  = np.array([
            [np.cos(kappa), -np.sin(kappa), 0, 0], 
            [np.sin(kappa),  np.cos(kappa), 0, 0],
            [0            ,              0, 1, 0], 
            [0            ,              0, 0, 1]
        ])
        self.Tn = np.eye(4)
        self.Tn[0, 3] = - self.Sx0
        self.Tn[1, 3] = - self.Sy0
            
        self.Tx = Q + np.diag([(brch.b_mu / (brch.Op if s > 0 else brch.Om))**2, -1, -1, brch.m_mu ** 2]).astype(np.float64)
        self.Q = self.Rn.dot(self.Tx).dot(self.Rn.T)
        #self.Q  = self.Tn.T.dot(self.Tx).dot(self.Tn)

    def fx(self, x, y, z): return FX(x, y, z, self.Q) 

## ---------- Quadrics on pencil lines ---------- #
#class TwoSheetLX(cfx_t):
#    def __init__(self, name, color, scale, Q, lmb, sign):
#        cfx_t.__init__(self, name, color, scale)
#        self.Q = Q
#
#        if sign > 0: self.lp, self.lm = lmb.LX0p, lmb.LY0p
#        else:        self.lp, self.lm = lmb.LX0m, lmb.LY0m
#        self.pts = 120
#        self.pl = lmb
#
#    def sxT(self, x, y, z):
#        return self.pl.Sx(x - self.lp, y -  self.lm)
#
#    def syT(self, x, y, z):
#        return self.pl.Sy(x - self.lp, y -  self.lm)
#
#    def szT(self, x, y, z):
#        return z
#
#    def fx(self, x, y, z): 
#        return FX(x, y, z, self.Q) 
#
## ---------- Quadrics on pencil lines ---------- #
#class TwoSheetL1(cfx_t):
#    def __init__(self, name, color, scale, Q, lmb, sign):
#        cfx_t.__init__(self, name, color, scale)
#        self.Q = Q
#        self.pts = 120
#        self.pl = lmb
#
#        self.lp = -lmb.lpy
#        self.lm = -lmb.lpx
#
#    def sxT(self, x, y, z):
#        return self.pl.Sx(x - self.lp, y -  self.lm)
#
#    def syT(self, x, y, z):
#        return self.pl.Sy(x - self.lp, y -  self.lm)
#
#    def szT(self, x, y, z):
#        return z
#
#    def fx(self, x, y, z): 
#        return FX(x, y, z, self.Q) 
#

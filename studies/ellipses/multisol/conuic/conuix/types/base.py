from conuix.base.constants import *
from conuix.base.atomics import *
from conuix.base.utils import *
import numpy as np
import math

def _branch(s, v1, v2): return v1 if s > 0 else v2

class basics_t:

    def __init__(self, bq, lp):
        self.cos = costheta(bq, lp)
        self.sin = (1 - self.cos**2)**0.5
        self.tan = self.sin / self.cos
        self.r   = lp.beta / bq.beta

        self.m_mu = lp.mass
        self.b_mu = lp.beta
        self.p_mu = lp.p
        self.e_mu = lp.e

        self.m_bq = bq.mass
        self.b_bq = bq.beta
        self.p_bq = bq.p

        self.bq = bq
        self.lp = lp

        # ------- parameters ------ #
        self.wp = omega(self, +1)
        self.wm = omega(self, -1)

        self.Op = Omega(self, +1)
        self.Om = Omega(self, -1)

        self.dp = delta(self, +1)
        self.dm = delta(self, -1)

        self.Gp = Gamma(self, +1)
        self.Gm = Gamma(self, -1)

        self.Ap = zA(self, +1)
        self.Am = zA(self, -1)

        self.Bp = zB(self, +1)
        self.Bm = zB(self, -1)

        self.Cp = zC(self, +1)
        self.Cm = zC(self, -1)

        self.D  = zD(self)
        self.E  = zE(self)

        self.Sx0  = Sx0(self)
        self.Sy0p = Sy0(self, +1)
        self.Sy0m = Sy0(self, -1)
        self.R_T = RT(self)

    def MQ(self, sg):                              return MQ(self, sg)
    def Sy0(self, sg):                             return _branch(sg, self.Sy0p, self.Sy0m) 
    def F_frame(self, nu = None):                  return F_frame(self, nu)
    def avec(self, sx, sy, z, c = 1):              return np.array([sx, sy, z, c])
    def nu4D(self, sx = 0, sy = 0, sz = 0, m = 0): return np.diag([0, 0, 0, m**2])
    def Q2(self, sx, sy, z, sg, m_nu): 
        vx = self.avec(sx, sy, z, 1)
        return vx.dot(MQ(self, sg) - self.nu4D(m = m_nu)).dot(vx.T)
    

class structs_t(basics_t):

    def __init__(self, bq, lp):
        basics_t.__init__(self, bq, lp)
   
    @property
    def delta_phi(self): return phi_delta(self)
    @property
    def dg_nu2_p(self): return degen_nu2(self, +1)
    @property
    def dg_nu2_m(self): return degen_nu2(self, -1)

    def to_Sx0(self, sx):     return sx + self.Sx0 
    def to_Sy0(self, sy, sg): return sy + self.Sy0(sg)

    def to_Sx(self, lp, lm): return SxL(self, lp, lm)
    def to_Sy(self, lp, lm): return SyL(self, lp, lm)

    def to_Lp(self, sx, sy): return LpS(self, sx, sy)
    def to_Lm(self, sx, sy): return LmS(self, sx, sy)

    def to_Lp0(self, lp, sg): return lp + L0(self, +1, sg)
    def to_Lm0(self, lm, sg): return lm + L0(self, -1, sg)
   
    def mW2(self, Sx, m_nu):      return mw2(self, Sx, m_nu)
    def mT2(self, Sx, Sy, m_nu):  return mt2(self, Sx, Sy, m_nu)

    def Z2(self, sx, sy, m_nu, sg): return Z2(self, sx, sy, m_nu, sg)
    def G2(self, sx, sy):           return G2(self, sx, sy)

    def x1(self, sx, sy, sg): return x1(self, sx, sy, sg)
    def y1(self, sx, sy, sg): return y1(self, sx, sy, sg)

    def BQ(self, sx, sy, m_nu2): return BQ(self, sx, sy, m_nu2)
    def LQ(self, sx, sy, m_nu2): return LQ(self, sx, sy, m_nu2)

    def Z2L(self, lp, lm, m_nu, sg):
        Lv = np.array([lp, lm])
        pr = LambdaN(self, sg)
        return Lv.dot(pr["N"].dot(Lv)) + pr["L"].dot(Lv) + pr["C"] - m_nu ** 2

    def vec_nu(self, sx, sy, z, chi, sg):
        x, y = Fnu_nx(self, sx, sy, z, chi, sg), Fnu_ny(self, sx, sy, z, chi, sg)
        z, e = Fnu_nz(self, z, chi),             Fnu_ne(self, sx, sy, z, chi, sg)
        return self.avec(x, y, z, e)

    def htilde(self, sx, sy, z, sg): return htilde(self, sx, sy, z, sg)




    # pencil second solution 
    def Pl2_m2(self, l, s1): return pl2_mass(self, l, s1)
    def Pt2_t(self, m_nu): return mt_nu(self, m_nu)
    def Pt2_m2(self, t): return m2nu_t(self, t)

    def Pl2(self, m_nu): 
        l1 = pl2_lambda(self, m_nu, +1, +1)
        l2 = pl2_lambda(self, m_nu, -1, -1)
        l3 = pl2_lambda(self, m_nu, +1, -1)
        l4 = pl2_lambda(self, m_nu, -1, +1)
        return [l1, l2, l3, l4]

    def to_Ap(self, sx, sy): return alphaK(self, sx, sy, +1, False)
    def to_Am(self, sx, sy): return alphaK(self, sx, sy, -1, False)

    def A_Sx(self, ap, am): return alphaK(self, ap, am, +1, True)
    def A_Sy(self, ap, am): return alphaK(self, ap, am, -1, True)

    def get_lines(self, m_nu):
        l0pp, l0mp = self.to_Lp0(0, +1), self.to_Lm0(0, +1)
        l0pm, l0mm = self.to_Lp0(0, -1), self.to_Lm0(0, -1)

        t = np.array([
            [l0pp, l0mp, self.Z2L(l0pp, l0mp, m_nu, +1)], 
            [l0pp, l0mp, self.Z2L(l0pp, l0mp, m_nu, -1)], 
            
            [l0pm, l0mp, self.Z2L(l0pm, l0mp, m_nu, +1)],
            [l0pm, l0mp, self.Z2L(l0pm, l0mp, m_nu, -1)],
            
            [l0pm, l0mm, self.Z2L(l0pm, l0mm, m_nu, +1)],
            [l0pm, l0mm, self.Z2L(l0pm, l0mm, m_nu, -1)]
        ])

        dk = []
        for i in range(len(t)): 
            for j in range(len(t)):
                if i >= j: continue
                l = t[i] - t[j]
                m = np.cross(t[i], t[j])
                d = l.dot(m)
                if abs(d) > 0: continue
                print(d)
                # Grassmann - Plucker relation:
                # D dot M = 0 
                s = 1 / sum(l**2) ** 0.5
                dk.append([l, t[i], t[j], l * s, m * s])
                print(l * s)
                print(m * s)
                print("____")
 


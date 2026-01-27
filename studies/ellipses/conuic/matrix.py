from atomics import *
import numpy
import math

def omegas(data):
    s, c = data.sin, data.cos 
    data.wp, data.wm = (1 / s) * (data.lep.b / data.jet.b - c), -(1 / s) * (data.lep.b / data.jet.b + c)
    data.op, data.om = (data.wp ** 2 + data.beta) ** 0.5,  (data.wm ** 2 + data.beta) ** 0.5

class SOL:

    def __init__(self, t, z, sx, sy, mw, mt, s):
        self.t  = t
        self.z2 = z
        self.sx = sx
        self.sy = sy
        self.mw = mw
        self.mt = mt
        self.sign = s

    def __str__(self):
        o  = string(self, "sign") + " "
        o +=       string(self, "sx") + " " + string(self, "sy")
        o += " " + string(self, "z2") + " "
        #o += " " + string(self, "mw") + " " + string(self, "mt")
        return o 

def eig(a, b, c, d):
    b = a *d
    c = a * d - c * d
    dsc = complex(b ** 2 - 4 * c)**2
    return (-b + dsc)/2, (-b - dsc) / 2

class matrix:
    def __init__(self): 
        self.cos  = costheta(self.jet, self.lep)
        self.sin  = (1 - self.cos**2) ** 0.5
        self.beta = self.lep.mass ** 2 / self.lep.e **2  #1 - self.lep.b ** 2
        omegas(self)

    def mW(self, sx):
        return complex(self.m_nu ** 2 - self.lep.mass ** 2 - 2 * self.lep.p * sx)**0.5

    def mT(self, sx, sy):
        a = self.m_nu ** 2 - self.lep.mass ** 2 + self.jet.mass ** 2
        return complex(a - 2 * self.lep.p * sx - 2 * self.jet.p * (self.cos * sx + self.sin * sy)) ** 0.5

    def _Z2(self, omega, Omega, sx, sy):
        o_inv = 1 / Omega ** 2
        a = o_inv - 1
        b = 2 * omega * o_inv
        c = -self.beta * o_inv
        d = 2 * self.lep.p
        e = self.lep.mass ** 2 - self.m_nu ** 2
        return a * sx ** 2 + sx * sy * b + c * sy ** 2 + d * sx + e 

    def Z2m(self, sx, sy):
        return self._Z2(self.wm, self.om, sx, sy)

    def Z2p(self, sx, sy):
        return self._Z2(self.wp, self.op, sx, sy)

    def _Sx0(self): return - self.lep.mass ** 2 / self.lep.p
    def _Sy0(self, omega): return - omega * self.lep.p / self.lep.b ** 2


    def tsolv(self):
        t1 = (self.wm * self.wp - self.beta + self.op * self.om) / (self.beta * (self.wm + self.wp))
        t2 = (self.wm * self.wp - self.beta - self.op * self.om) / (self.beta * (self.wm + self.wp))
        return t1, t2

    def getSolutions(self):
        def _SxSy(omega, Omega, t, s):
            a = (1 / Omega**2 -1 ) + (2 * omega / Omega**2)*t + (omega**2 / Omega ** 2 - 1) * t ** 2
            b = 2 * self.lep.p 
            c = self.lep.mass ** 2 - self.m_nu ** 2 
            dsc = complex(b ** 2 - 4 * a * c) ** 0.5 
            sx1, sx2 = (-b + dsc) / (2 * a), (-b - dsc)/(2 * a)
            sy1, sy2 = sx1 * t, sx2 * t
            z21, z22 = self._Z2(omega, Omega, sx1, sy1), self._Z2(omega, Omega, sx2, sy2)
            s1, s2 = SOL(t, z21, sx1, sy1, self.mW(sx1), self.mT(sx1, sy1), s), SOL(t, z22, sx2, sy2, self.mW(sx2), self.mT(sx2, sy2), s)
            return [s1, s2]

        t1, t2 = self.tsolv()
        p = _SxSy(self.wp, self.op, t1, "+1") + _SxSy(self.wp, self.op, t2, "+2")
        m = _SxSy(self.wm, self.om, t1, "-1") + _SxSy(self.wm, self.om, t2, "-2")
        return p + m 
    
    def tan2psi(self):
        a = (self.beta * self.sin**2 - self.cos**2) * self.jet.b ** 2 + abs(1 - self.beta)
        return - a / (self.jet.b ** 2 * self.cos * self.sin * (1 + self.beta))

    def psi(self):
        a = (self.beta * self.sin**2 - self.cos**2) * self.jet.b ** 2 + abs(1 - self.beta)
        return math.atan(- a / (self.jet.b ** 2 * self.cos * self.sin * (1 + self.beta))) * 0.5

    def DG2_eig(self):
        a = self.cos / self.sin
        b = (1 + self.tan2psi() ** 2)**0.5
        return a * (1 - self.beta + (1 + self.beta)*b), a * (1 - self.beta - (1 + self.beta)*b)

    def SxSy(self, tau, z):
        l1, l2 = self.DG2_eig()
        kappa = ((self.op * self.om)**2 / (self.wp - self.wm))**0.5
        a, b = (kappa * z/ l1)**0.5, (kappa * z/ l2)**0.5
        tpsi = self.psi()
        sx = (a * math.cos(tpsi) * cosh(tau) - b * math.sin(tpsi) * sinh(tau))
        sy = (a * math.sin(tpsi) * cosh(tau) + b * math.cos(tpsi) * sinh(tau))
        return sx, sy 

    def GetTauZ(self, sx, sy):
        l1, l2 = self.DG2_eig()
        kappa = ((self.op * self.om)**2 / (self.wp - self.wm))**0.5
        tpsi = self.psi()
        a, b = (kappa/ l1)**0.5, (-kappa/ l2)**0.5
        X =   sx * math.cos(tpsi) + sy * math.sin(tpsi) 
        Y = - sx * math.sin(tpsi) + sy * math.cos(tpsi)
        R = a * Y / (b * X) 
        tau = atanh(R if abs(R) < 1 else 1 / abs(R))
        print(tau)
        k = (l1 * X**2 + l2 * Y**2) / kappa
        return tau, k


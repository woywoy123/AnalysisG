from atomics import *
from mobius import *
import numpy as np
import cmath 

def roots(a, b, c):
    q = complex(b ** 2 - 4 * a * c) ** 0.5
    t1 = (- b + q) / (2 * a)
    t2 = (- b - q) / (2 * a)
    return t1, t2   


class coef_t: 
    def __init__(self):
        self.a = None
        self.b = None
        self.c = None
        self.d = None
        self.e = None

    def __str__(self):
        o = ""
        if self.a is not None: o += string(self, "a") + " "
        if self.b is not None: o += string(self, "b") + " "
        if self.c is not None: o += string(self, "c") + " "
        if self.d is not None: o += string(self, "d") + " "
        if self.e is not None: o += string(self, "e") + " "
        return o


class branching:
    def QT(self, t): 
        return 2 * t / (self.wp + self.wm) - 1

    def TI(self): 
        a = -(1 - self.lep.b ** 2 - self.wm * self.wp)
        k = self.Om * self.Op
        d = (1 - self.lep.b ** 2) * (self.wm + self.wp)
        return (a + k)/d, (a - k)/d

    def Sx(self, t): 
        a = - self.lep.p 
        k = complex(self.lep.p ** 2 - self.QT(t) * (self.lep.mass ** 2 - 0))**0.5
        return (a + k)/self.QT(t), (a - k)/self.QT(t)

    def __init__(self, data):
        self.jet  = data.jet
        self.lep  = data.lep
        self.r    = self.lep.b / self.jet.b

        # ----- Kinematics ----- #
        b_mu      = self.lep.b

        # --------- Define a symmetry axis i.e. fx(psi) -> fx(-psi)
        self.tapsi = (1 - data.cos) / data.sin # tan(psi)
        self.copsi = (1 + data.cos) / data.sin # cot(psi)

        self.kp = (self.r + 1) 
        self.km = (self.r - 1)

        self.wp =  (1 / 2) * (self.kp * self.tapsi + self.km * self.copsi)
        self.wm = -(1 / 2) * (self.km * self.tapsi + self.kp * self.copsi)
       
        self.Op = (self.wp ** 2 + 1 - b_mu**2) ** 0.5
        self.Om = (self.wm ** 2 + 1 - b_mu**2) ** 0.5
        self.oo = self.Op * self.Om

        # ------ hyperbolic axis ------ #
        self.atn = (b_mu * ( self.Op * self.copsi - self.Om * self.tapsi )) / ( b_mu ** 2 - self.oo)
       
        self._Zp = coef_t()
        self._Zp.a = - (self.wp ** 2 - b_mu ** 2) / self.Op ** 2
        self._Zp.b = 2 * self.wp / self.Op ** 2
        self._Zp.c = - (1 - b_mu ** 2) / self.Op ** 2
        self._Zp.d = 2 * self.lep.p
        self._Zp.e = (self.lep.mass ** 2 - data.m_nu ** 2)

        self._Zm = coef_t()
        self._Zm.a = - (self.wm ** 2 - b_mu ** 2) / self.Om ** 2
        self._Zm.b = 2 * self.wm / self.Om ** 2
        self._Zm.c = - (1 - b_mu ** 2) / self.Om ** 2
        self._Zm.d = 2 * self.lep.p
        self._Zm.e = (self.lep.mass ** 2 - data.m_nu ** 2)

        # ----------- Interference conic sections ---------- #
        # let t = Sy / Sx
        self._Zi = coef_t()
        self._Zi.a =  4 * b_mu * data.cos * b_mu / (self.jet.b * (data.sin * self.oo) ** 2)
        self._Zi.b =  4 * b_mu * ( 1 - b_mu ** 2 - self.wp * self.wm) / (self.jet.b * data.sin * (self.oo)**2)
        self._Zi.c = -4 * b_mu * data.cos * (1 - b_mu**2) / (self.jet.b * (data.sin * self.oo)**2) 

        t1, t2 = self.TI()
        self._Zi.t1 = t1; self._Zi.t2 = t2
        
        sx1, sx2 = self.Sx(t1)
        self._Zi.dS  = [[sx1, sx1 * t1], [sx2, sx2 * t1]]

        sx1, sx2 = self.Sx(t2)
        self._Zi.dS += [[sx1, sx1 * t2], [sx2, sx2 * t2]]

        self.eig = coef_t()
        mx = complex(b_mu ** 4 * data.cos ** 2 + data.sin * (self.oo)**2)**0.5
        self.eig.a = 2 * b_mu / (self.jet.b * (data.sin * self.oo)**2) * (b_mu ** 2 * data.cos + mx)
        self.eig.b = 2 * b_mu / (self.jet.b * (data.sin * self.oo)**2) * (b_mu ** 2 * data.cos - mx)
        
        y = b_mu **2 * (2 - self.jet.b ** 2) - self.jet.b ** 2 * (2 - b_mu ** 2) * (data.cos**2 - data.sin**2)
        x = (self.jet.b ** 2 * (2 - self.jet.b ** 2) * (2 * data.sin * data.cos))

        print("->", -(self.oo * (self.tapsi * self.copsi))**0.5, self.wp * self.wm, self.wp, self.wm)


        self.eig.c = math.tan(atan2(y, x) / 2)
        self.eig.d = atanh(((self.Op / b_mu) * self.copsi - (self.Om / b_mu) * self.tapsi) * b_mu / ( b_mu**2 - self.oo ))

    def Zm(self, sx = None, sy = None):
        if sx is not None and sy is not None:  
            return self._Zm.a * sx ** 2 + self._Zm.b * sx * sy + self._Zm.c * sy ** 2  + self._Zm.d * sx + self._Zm.e 
        return [[i[0], i[1], self.Zm(i[0], i[1])] for i in self._Zi.dS]

    def Zp(self, sx = None, sy = None):
        if sx is not None and sy is not None:  
            return self._Zp.a * sx ** 2 + self._Zp.b * sx * sy + self._Zp.c * sy ** 2  + self._Zp.d * sx + self._Zp.e 
        return [[i[0], i[1], self.Zp(i[0], i[1])] for i in self._Zi.dS]

class matrix:
    def __init__(self):
        self.RT = None

    def cache(self):
        rbl = self.lep.b / self.jet.b

        self.cos  = costheta(self.jet, self.lep)
        self.sin  = (1 - self.cos**2) ** 0.5
        self.branch = branching(self)

        self.tpsi = (rbl - self.cos) / self.sin
        self._tpsi = (-rbl - self.cos) / self.sin


        self.cpsi = 1.0 / (1 + self.tpsi**2)**0.5
        self.spsi = self.tpsi * self.cpsi

        self.o    = (self.tpsi ** 2 + 1 - self.lep.b ** 2)**0.5
       
        self._Z2   = coef_t()
        self._Z2.a = (1 - self.o ** 2)/(self.o ** 2)
        self._Z2.b = (2.0 * self.tpsi)/(self.o ** 2)
        self._Z2.c = (self.tpsi ** 2 -  self.o ** 2)/(self.o ** 2)
        self._Z2.d = 2.0 * self.lep.p
        self._Z2.e = self.lep.mass ** 2 - self.m_nu ** 2

        self._Sx   = coef_t()
        self._Sx.a = (self.o / self.lep.b) * self.cpsi 
        self._Sx.b = self.spsi 
        self._Sx.c = - self.lep.mass ** 2 / self.lep.p

        self._Sy   = coef_t()
        self._Sy.a = (self.o / self.lep.b) * self.spsi 
        self._Sy.b = self.cpsi 
        self._Sy.c = - self.tpsi * self.lep.e / self.lep.b

        self._x1   = coef_t()
        self._x1.a = self.lep.p
        self._x1.b = 1 / self.o
        self._x1.c = self.lep.b * self.cpsi
        self._x1.d =     self.o * self.spsi

        self._y1   = coef_t()
        self._y1.a = 1 / self.o
        self._y1.b =     self.o * self.cpsi
        self._y1.c = self.lep.b * self.spsi

        self._tZ   = coef_t()
        self._tZ.a = self.o / self.lep.b
        self._tZ.b = self.lep.mass ** 2 / self.lep.p
        self._tZ.c = (self.lep.e / self.lep.b ** 2) * self.tpsi

















        # untested
        self._mass   = coef_t()
        self._mass.a = - self.lep.mass ** 2
        self._mass.b = - 2 * self.lep.p
        self._mass.c =   self.jet.mass ** 2
        self._mass.d = - 2 * self.jet.b

        self._line   = coef_t()
        self._line.a = - 0.5 / self.lep.p
        self._line.b =   0.5 * (self.lep.mass ** 2 / self.lep.p)
        self._line.c = - 0.5 / (self.jet.p * self.sin)
        self._line.d =   0.5 / (self.jet.p * self.sin) + self.cos * 0.5 / (self.lep.p * self.sin)

        self._line.e  =  0.5 * (self.jet.mass ** 2 / (self.jet.p * self.sin)) 
        self._line.e +=  0.5 * self.lep.mass ** 2 * self.cos / (self.lep.p * self.sin) 
        self._line.e +=  (self.lep.e / self.lep.b) * self.tpsi

        self._rmW   = coef_t()
        self._rmW.a = - 1.0 / (2.0 * self.lep.p)
        self._rmW.b = - self.lep.mass ** 2 / (2 * self.lep.p)
        self._rmW.c =  1.0 / (2 * self.sin) * (1.0 / self.jet.p + self.cos / self.lep.p)
        self._rmW.d =  1.0 / (2 * self.sin) * (self.jet.mass ** 2 / self.jet.p + self.lep.mass ** 2 * self.cos / self.lep.p)
        self._rmW.e = -1.0 / (2 * self.sin * self.jet.p)


        self.HBX  = nulls(3, 3)
        self.HBX[0][0] = 1.0 / self.o
        self.HBX[1][0] = self.tpsi / self.o
        self.HBX[2][0] = 0

        self.HBX[0][1] = 0
        self.HBX[1][1] = 0
        self.HBX[2][1] = 1

        self.HBX[0][2] = 0
        self.HBX[1][2] = 0
        self.HBX[2][2] = 0
        self.HBX = np.array(self.HBX)
        self.HTX = self.R_T.dot(self.HBX)


        self.HBC  = nulls(3, 3)
        self.HBC[0][0] = 0 
        self.HBC[1][0] = 0 
        self.HBC[2][0] = 0

        self.HBC[0][1] = 0
        self.HBC[1][1] = 0
        self.HBC[2][1] = 0

        self.HBC[0][2] = self.lep.b * self.cpsi / self.o
        self.HBC[1][2] = self.lep.b * self.spsi / self.o
        self.HBC[2][2] = 0

        self.HBC = np.array(self.HBC)
        self.HTC = self.R_T.dot(self.HBC)

        self.HBS  = nulls(3, 3)
        self.HBS[0][0] = 0 
        self.HBS[1][0] = 0 
        self.HBS[2][0] = 0

        self.HBS[0][1] = 0
        self.HBS[1][1] = 0
        self.HBS[2][1] = 0

        self.HBS[0][2] =  self.spsi 
        self.HBS[1][2] = -self.cpsi 
        self.HBS[2][2] = 0

        self.HBS = np.array(self.HBS)
        self.HTS = self.R_T.dot(self.HBS)


    def test(self, tau):
        self.debug()


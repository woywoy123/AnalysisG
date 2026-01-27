from relations import *
from atomics import *
import math

class abstract:
    def __init__(self, jet, lep, m_nu):
        self.cos   = costheta(jet, lep)   
        self.sin   = (1 - self.cos**2) ** 0.5
        self.theta = math.acos(self.cos)

        self.m_nu  = m_nu
        self.lep   = lep
        self.jet   = jet

        self.j_b = (complex(1 - (jet.mass / jet.e)**2) ** 0.5).real
        self.l_b = (complex(1 - (lep.mass / lep.e)**2) ** 0.5).real

        self.j_eta = np.acosh(jet.e / jet.mass)
        self.l_eta = np.acosh(lep.e / lep.mass)

    def __str__(self):
        o = ""
        for i in self.__dict__: o += "key: " + str(i) + " val: " + str(self.__dict__[i]) + " | "
        return o

class Pencils(abstract):
    def __init__(self, jet, lep, m_nu, br):
        abstract.__init__(self, jet, lep, m_nu)

        # -------- Constants for Z^2 branches -------- #
        self.w = omega(self.l_eta, self.j_eta, self.sin, self.theta, br)
        self.o = complex(self.w ** 2 + 1 - self.l_b ** 2)**0.5

        # -------- Quadric Properties --------- #
        # - Coefficients 
        print(Z2(self))

        # - Centers:
        self.Sx0 = - self.lep.mass ** 2 / self.lep.p
        self.Sy0 = - self.w * self.lep.e ** 2 / self.lep.p
        self.SxC = np.array([1, - 1 / (np.tan(self.theta) * (1 - self.l_b**2))]) * self.Sx0

        # - eigenvalues:
        self.l1 = (self.l_b / self.o)**2
        self.l2 = -1

        # - rotation
        self.psi = np.atan(self.w) * 0.5
        self.RV = np.array([
            [np.cos(self.psi), -np.sin(self.psi)], 
            [np.sin(self.psi),  np.cos(self.psi)]
        ]) 
 
    def SxSy(self, Sx, Sy):
        pass 




class Interference(abstract):
    def __init__(self, zp, zm, cx):
        self.wm, self.wp,  self.om, self.op  = zm.w, zp.w, zm.o, zp.o
        self.lb, self.bb, self.sin, self.cos = zp.l_b, zp.j_b, zp.sin, zp.cos
        self.tan = self.sin / self.cos

        # ------ Delta Constants -------- #
        self.Gp = GammaR(self, +1)
        self.Dp = deltaR(self, +1)

        self.Gm = GammaR(self, -1)
        self.Dm = deltaR(self, -1)

        self.psip = np.atan(self.Dp)
        self.psim = np.atan(self.Dm)

        self.phi  = -(self.psip + self.psim) * 0.5
        self.phi_ =  (self.psip - self.psim) * 0.5 

        self.RN = self.Gp * self.Gm / ( np.cos(self.psip) * np.cos(self.psim) )
        self.lp = -self.RN * (np.cos(self.phi_) ** 2)
        self.lm =  self.RN * (np.sin(self.phi_) ** 2)

        self.RV = np.array([
            [np.cos(self.phi), -np.sin(self.phi)], 
            [np.sin(self.phi),  np.cos(self.phi)]
        ]) 
          
        # ----- root1 + root2 ------ #
        self.d1_p_d2 = - 2 * ( 1 - self.lb ** 2 - self.wm * self.wp) / (self.wp + self.wm)

    def SxSy(self, K, tau):   return self.RV.dot(Hypers(self, tau, K).T)
    def DeltaG(self, Sx, Sy): return - self.Gp * self.Gm * (Sx - self.Dp * Sy)*(Sx - self.Dm * Sy)

class NuConuix:
    def __init__(self, lep, bqrk, nu = None): 
        self.RT = None
        self.lep = lep
        self.jet = bqrk
        self._p = Pencils(bqrk, lep, nu.mass if nu is not None else 0, +1)
        self._m = Pencils(bqrk, lep, nu.mass if nu is not None else 0, +1)
        self.pm = Interference(self._p, self._m, self)

        self.cols  = ["blue", "navy", "cyan", "green"]
        self.cols += ["blue", "navy", "cyan", "green"]

        self.style  = ["-", "-", "-", "-"]
        self.style += ["-.", "-.", "-.", "-."]

    def switch(self, sign): return self._p if sign > 0 else self._m

    def Z2(self, Sx, Sy, sign): 
        data = self.switch(sign)
        return Z2(data, Sx, Sy)

    def mW(self, Sx, sign): 
        data = self.switch(sign)
        return mW(data.m_nu, data.lep, Sx)

    def mT(self, Sx, Sy, sign): 
        data = self.switch(sign)
        return mT(data.m_nu, data.lep, data.jet, data.sin, data.cos, Sx, Sy)

    def x1(self, Sx, Sy, sign):
        data = self.switch(sign)
        return x1(Sx, Sy, data.w, data.o)

    def y1(self, Sx, Sy, sign):
        data = self.switch(sign)
        return y1(Sx, Sy, data.w, data.o)

    def H_tilde(self, Z, Sx, Sy, sign):
        data = self.switch(sign)
        return H_tilde(Z, Sx, Sy, data)

    def Sx(self, tau, sign): 
        data = self.switch(sign)
        return SxHyper(tau, data) 

    def Sy(self, tau, sign): 
        data = self.switch(sign)
        return SyHyper(tau, data) 

    def SxP(self, tau, sign):
        return self.pm.SxP(tau, sign)

    def SyP(self, tau, sign):
        return self.pm.SyP(tau, sign)

    def DeltaG2(self, Sx, Sy, alt = True):
        if alt: return DeltaG2_alt(Sx, Sy, self._p, self._m) 
        return DeltaG2(Sx, Sy, self._p, self._m) 

    @property 
    def R_T(self):
        if self.RT is not None: return self.RT
        px, py, pz = self.lep.px, self.lep.py, self.lep.pz
        phi   = np.arctan2(py, px)
        theta = np.arctan2(np.sqrt(px**2 + py**2), pz)
        R_z   = rotation_z(-phi)
        R_y   = rotation_y(0.5*np.pi - theta)
        
        b_vec = np.array([self.jet.px, self.jet.py, self.jet.pz])
        b_rot = R_y @ (R_z @ b_vec)
        R_x = rotation_x(-np.arctan2(b_rot[2], b_rot[1]))
        self.RT = R_z.T @ R_y.T @ R_x.T
        return self.RT


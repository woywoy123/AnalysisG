from atomics import *
from debug import *
import numpy as np
import cmath
import math

class branches:
    def __init__(self, var, sign, eps = -1):
        self.m_nu = 1
        self.l1 = -1 
        self.eps = eps
        self.l2 = (var.b_mu / signs(var.op, var.om, sign)) ** 2
        self.o = signs(var.op, var.om, sign)
        self.w = signs(var.wp, var.wm, sign)

        self.kappa = math.atan(signs(var.wp, var.wm, sign))
        self.Sx0   = - var.m_mu **2 / var.p_mu
        self.Sy0   = - math.tan(self.kappa) * var.lep.e / var.lep.b
        self.S0    = np.array([[self.Sx0], [self.Sy0]])

        self.eigv = np.array([
            [self.o ,         0], 
            [0      , -var.b_mu]
        ]) * 1 / var.b_mu

        self.Rot = np.array([
            [math.cos(self.kappa), - math.sin(self.kappa)],
            [math.sin(self.kappa),   math.cos(self.kappa)]
        ])

        self.M = self.Rot.dot(self.eigv)
        self.sign = sign
        self.inst = var

    def y1x1(self, tau, m_nu = None):
        if m_nu is None: m_nu = self.inst.mass_neutrino(self.sign)
        Sy, Sx = self.Sy(tau), self.Sx(tau)
        x1 = self.inst.x1(Sx, Sy, self.sign)
        y1 = self.inst.y1(Sx, Sy, self.sign)
        z1 = self.inst.Z2(Sy, Sx, m_nu, self.w, self.o)
        return x1, y1, abs(z1.real)**0.5
   
    def Sx(self, tau, m_nu = None):
        if m_nu is None: m_nu = self.m_nu
        Sx  = math.cos(self.kappa) * math.cosh(tau) / (self.l2**0.5) - self.l1 * math.sin(self.kappa) * math.sinh(tau)
        return self.eps * m_nu * Sx + self.Sx0

    def Sy(self, tau, m_nu = None):
        if m_nu is None: m_nu = self.m_nu
        Sy = math.sin(self.kappa) * math.cosh(tau) / (self.l2**0.5) + self.l1 * math.cos(self.kappa) * math.sinh(tau)
        return self.eps * m_nu * Sy + self.Sy0

    def SxSy(self, tau, m_nu = None):
        if m_nu is None: m_nu = self.inst.mass_neutrino(self.sign)
        S = self.M.dot(np.array([[math.cosh(tau), math.sinh(tau)]]).T)
        S = (m_nu * self.eps * S + self.S0).reshape((-1))
        return S

    def tau(self, Sx, Sy):
        S = np.array([[Sx], [Sy]])
        S = np.linalg.inv(self.M.T).T.dot(S - self.S0).reshape(2)
        S = ((S[1] / S[0])) 
        if abs(S) > 1: S = 1 / S; 
        return math.atanh( S )

class hyper:
    def __init__(self, instance, sign):
        self.inst = instance
        self.sign = sign
        self.B = np.array(self.y1x1(0))

    def y1x1(self, tau):
        sy, sx = self.inst.dG2_SySx(tau)
        x1 = self.inst.x1(sx, sy, self.sign)
        y1 = self.inst.y1(sx, sy, self.sign)
        w = signs(self.inst.wp, self.inst.wm, self.sign)
        o = signs(self.inst.op, self.inst.om, self.sign)
        z1 = self.inst.Z2(sy, sx, self.inst.mass_neutrino(self.sign), w, o)
        return x1, y1, abs(z1)**0.5

    def SxSy(self, tau): 
        sy, sx = self.inst.dG2_SySx(tau)
        return np.array([self.sign * sx, self.sign * sy, self.inst.dG2(self.sign * sy, self.sign * sx)])

    def tau(self, Sx, Sy):
        S = np.array([[Sx, Sy]]).T
        S = np.linalg.inv(self.inst.eigm).dot(S)
        S = S.T.reshape(2)
        inv = int(abs(S[0]) < abs(S[1]))
        r = S[0] / S[1] if inv else S[1] / S[0]
        return math.atanh(r)

class line:

    def __init__(self, m, b, x0 = None, y0 = None, name = None, var = None, sign1 = None, sign2 = None):
        self.m = m
        self.b = b
        self.name = name
        self.var = var
        self.sol_pts = None
        self.sign1 = sign1
        self.sign2 = sign2
        self.x0 = x0
        self.y0 = y0
        self.rz = 0 
        self.ry = 0
        self.rx = 0

        if x0 is not None and y0 is not None: 
            sx, sy, sz = var.SolveSxSy(x0, y0, var.mass_neutrino(sign1), sign2)
            self.sol_pts = np.array([x0, y0, (abs(sz)**0.5).real])
            return 
        if x0 is not None and y0 is None: self.sol_pts = np.array([x0, m * x0 + b, 1])
        if y0 is not None and x0 is None: self.sol_pts = np.array([(y0 - b)/m, y0, 1]) 
        if sum([i is not None for i in [x0, y0]]) == 2: self.sol_pts = np.array([x0, y0, 1])

    @property
    def Rz(self):
        rz = np.array([
            [ math.cos(self.rz), -math.sin(self.rz),  0],
            [ math.sin(self.rz),  math.cos(self.rz),  0],
            [                 0,         0         ,  1]
        ])
        return rz

    @property
    def Ry(self):
        ry = np.array([
            [  math.cos(self.ry),         0,  math.sin(self.ry)],
            [ 0                 ,         1,                  0],
            [- math.sin(self.ry),         0,  math.cos(self.ry)]
        ])
        return ry


    @property
    def Rx(self):
        rx = np.array([
            [ 1,                  0,                  0],
            [ 0,  math.cos(self.rx), -math.sin(self.rx)],
            [ 0,  math.sin(self.rx),  math.cos(self.rx)]
        ])
        return rx





    def __call__(self, t):
        Sx, Sy, z1 = self.var.SolveSxSy(self.x0, self.y0, self.var.mass_neutrino(self.sign1), self.sign2)
        return np.array([t, self.m * t + self.b, z1.real])

    def __hash__(self): return sum([hash(self.m) + hash(self.b)])

    def __eq__(self, other):
        if other.__class__ == self.__class__: pass
        else: return False
        return False

    def __str__(self):
        s = "y = " + str(self.m) + " x " + ("+" if self.b > 0 else "") + str(self.b)
        if self.sol_pts is None: return s
        s += " -> (x0, y0) = (" + str(self.sol_pts[0]) + ", " + str(self.sol_pts[1]) + ")"
        return s

class variables(debug):

    def __init__(self, lep, jet):
        debug.__init__(self)
        self.p_mu = lep.e * lep.b
        self.p_b  = jet.e * jet.b

        self.b_mu = lep.b
        self.b_b  = jet.b

        self.m_mu = lep.mass
        self.m_b  = jet.mass

        self.c_th = costheta(lep, jet)
        self.theta = math.acos(self.c_th)
        self.s_th = (1 - self.c_th ** 2) ** 0.5
        self.t_th = self.s_th / self.c_th

        self.wp = self.omega(+1); self.wm = self.omega(-1.0)
        self.op = self.Omega(+1); self.om = self.Omega(-1.0)

        self.dp, self.Gp = self.delta(+1), self.Gamma(+1.0)
        self.dm, self.Gm = self.delta(-1), self.Gamma(-1.0)

        self.dG2_factorization()

        # --------- Branch specific parameterization ---------- #
        self.Z2_p = branches(self, +1)
        self.Z2_m = branches(self, -1)

        self.Z2_pp = branches(self, +1, +1)
        self.Z2_mp = branches(self, -1, +1)
        self.Z2_pm = branches(self, +1, -1)
        self.Z2_mm = branches(self, -1, -1)

        self.G2_p = hyper(self, +1)
        self.G2_m = hyper(self, -1)

    def Z2_crootL(self, sign, raw = False):
        m_nu = self.mass_neutrino(sign)
        tn = signs(self.t_psi_p, self.t_psi_m, sign)
        w = signs(self.wp, self.wm, sign)
        o = signs(self.op, self.om, sign)
        kx = self.Z2(0, 0, m_nu, w, o, True)
        a, b, c, bx, cx = kx
        ax = a + b / tn + c / tn**2
        dsc = complex(abs(bx**2 - 4 * ax * cx))**0.5
        if abs(dsc.imag) > 0: return []
        dsc = dsc.real
        SxP, SxM = (-bx + dsc)/(2 * ax), -(bx + dsc)/(2 * ax)
        if raw: return [SxP, SxM]
        
        o  = []
        for i in [[+1, +1], [+1, -1], [-1, +1], [-1, -1]]:
            for k in [[SxP, SxP], [SxP, SxM], [SxM, SxP], [SxM, SxM]]:
                o += self.clines(k[0], k[1], sign, sign, i[0], i[1])[-1]
        return o

    def lines(self, Sx11 = None, Sy11 = None, Sx22 = None, Sy22 = None):
        # --------- The plus branch ------------- #
        if Sx11 is None: Sx11,  Sy11 = self.dZ2dSxSy_max(+1)
        px11, py11  = self.x1(Sx11, Sy11, +1), self.y1(Sx11, Sy11, +1)
        px12, py12  = self.x1(Sx11, Sy11, -1), self.y1(Sx11, Sy11, -1)

        # ---- Build lines ->  y = mx + b
        gr_p = (py12 - py11)/(px12 - px11)
        bp   = py12 - gr_p * px12

        # --------- The minus branch ------------- #
        if Sx22 is None: Sx22,  Sy22 = self.dZ2dSxSy_max(-1)
        mx11, my11  = self.x1(Sx22, Sy22, +1), self.y1(Sx22, Sy22, +1)
        mx12, my12  = self.x1(Sx22, Sy22, -1), self.y1(Sx22, Sy22, -1)

        # ---- Build lines ->  y = mx + b
        gr_m = (my12 - my11)/(mx12 - mx11)
        bm   = my12 - gr_m * mx12

        # -------- Finding line of intersection ---------- #
        x0  = (bm - bp)/(gr_p - gr_m)
        return {
            "plus"  : {"m" : gr_p, "b" : bp, "x0" : x0, "y0" : gr_p * x0 + bp},
            "minus" : {"m" : gr_m, "b" : bm, "x0" : x0, "y0" : gr_m * x0 + bm},
        }


    def clines(self, Sx11, Sx22, sign1, sign2, sign3, sign4):
        # -------- y = mx + b ------------- #
        def PXX(x11, y11, x12, y12, x21, y21, x22, y22):
            u = [x12 - x11, y12 - y11]
            v = [x22 - x21, y22 - y21]
            return u[0] * v[1] - u[1] * v[0]
        
        def Mxx(x1, x2, y1, y2): return (y2 - y1) / (x2 - x1)
        def Bxx(x1, y1, mxx):    return y1 - mxx * x1
        def Ixx(b1, b2, m1, m2): 
            x0  = (b1 - b2)/(m2 - m1)
            y0, y1 = m1 * x0 + b1, m2 * x0 + b2
            if abs(y0 - y1) ** 2 > 1e-12: return False, False
            return x0, y0

        def MxB(x11, x12, y11, y12, x22, x21, y22, y21, sign1, sign2): 
            m12, m21 = Mxx(x11, x12, y11, y12), Mxx(x21, x11, y21, y11)
            m11, m22 = Mxx(x11, x22, y11, y22), Mxx(x22, x11, y22, y11)

            b11, b12 = Bxx(x11, y11, m11), Bxx(x12, y12, m12)
            b21, b22 = Bxx(x21, y21, m21), Bxx(x22, y22, m22)

            x0, y0 = Ixx(b11, b12, m11, m12)
            f = [
                    line(m11, b11, x0, y0, "11", self, sign1, sign1), 
                    line(m12, b12, x0, y0, "12", self, sign1, sign2), 
                    line(m21, b21, x0, y0, "21", self, sign2, sign1), 
                    line(m22, b22, x0, y0, "22", self, sign2, sign2)
            ]

            o = []
            for i in f: o += [i] if i not in o else []
            return o

        tn11 = signs(self.t_psi_p, self.t_psi_m, sign1)
        tn22 = signs(self.t_psi_p, self.t_psi_m, sign2)

        px11, py11 = self.x1(Sx11, Sx11 / tn11, sign3), self.y1(Sx11, Sx11 / tn11, sign3)
        px12, py12 = self.x1(Sx11, Sx11 / tn11, sign4), self.y1(Sx11, Sx11 / tn11, sign4)
                                                                                        
        px21, py21 = self.x1(Sx22, Sx22 / tn22, sign3), self.y1(Sx22, Sx22 / tn22, sign3)
        px22, py22 = self.x1(Sx22, Sx22 / tn22, sign4), self.y1(Sx22, Sx22 / tn22, sign4)

        # -------- Check if any of the points are collinear -> skip.
        if PXX(px11, py11, px12, py12, px21, py21, px22, py22) == 0: return False, {}
        d = MxB(px11, px12, py11, py12, px22, px21, py22, py21, sign1, sign2)
        return True, d


    def dZ2dSxSy_max(self, sign):
        tn = signs(self.t_psi_p, self.t_psi_m, sign)
        return (self.p_mu * tn) / (tn + self.t_th), self.p_mu / (tn + self.t_th)



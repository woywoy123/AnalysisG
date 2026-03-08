import matplotlib.pyplot as plt
from original import NuSol
import numpy as np
import math

from particle import *
from atomics import * 
from figures import *
from base import *

     
class algorithm:
    def __init__(self, jet, lep, nu = None):
        self.lep = lep 
        self.jet = jet
        self.nu  = nu
        self.ref = NuSol(self.jet, self.lep, self.nu)
        self.data = data_t(jet, lep)

        self.Z2_pp = Z2_t(self.data, +1, +1)
        self.Z2_pm = Z2_t(self.data, +1, -1)
        self.Z2_mp = Z2_t(self.data, -1, +1)
        self.Z2_mm = Z2_t(self.data, -1, -1)
        self.G2    = G2_t(self.data)

        self.plot_line   = False
        self.plot_ellipse = True
        self.Build_lines()
        self.Ellipses() 
        self.debug()

    def Build_lines(self):
        m_nu_pp = nu_mass(self.data, +1, +1) # delta +, omega +
        m_nu_mp = nu_mass(self.data, -1, +1) # delta -, omega + 
        m_nu_pm = nu_mass(self.data, +1, -1) # delta +, omega -
        m_nu_mm = nu_mass(self.data, -1, -1) # delta -, omega -

        S_pp = AQuad(self.data, +1, +1, m_nu_pp) # delta +, omega +
        S_mp = AQuad(self.data, -1, +1, m_nu_pm) # delta -, omega + 
        S_pm = AQuad(self.data, +1, -1, m_nu_mp) # delta +, omega -
        S_mm = AQuad(self.data, -1, -1, m_nu_mm) # delta -, omega -

        MX = {
                "++++": S_pp["SP"], "+++-": S_pp["SM"], 
                "++-+": S_pm["SP"], "++--": S_pm["SM"], 
                "+-++": S_mp["SP"], "+-+-": S_mp["SM"],
                "---+": S_mm["SP"], "----": S_mm["SM"]
        }

        f = ["+", "-"]
        K = [f[i] + f[j] + f[k] + f[t] for i in range(2) for j in range(2) for k in range(2) for t in range(2)]
        M = {"[" + i + "] -> [" + j + "]" : [MX[i], MX[j]] for i in K for j in K if i in MX and j in MX}
        self.D = {}
        for i in M:
            r = line_t(M[i][0], M[i][1], i)
            if not r.Intersect: continue
            if r.Collinear:     continue
            self.D[i] = r 

        if not self.plot_line: return 
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes()
        ax.set_xlabel(f"Sx")
        ax.set_ylabel(f"Sy")
        ax.grid(True, alpha=0.7, linestyle=':')

        # Add (0,0) crosshairs for visual reference
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Keep equal aspect ratio
        ax.axis('equal')

        t = np.linspace(-100000, 10000000000, 100000)
        for i in self.D:
            r = self.D[i](t)
            ax.plot(t, r, linestyle ="-", label = i)
            
        sx, sy = self.ref.Sx, self.ref.Sy
        ax.scatter(sx, sy, marker="x", s = 80, label = "truth", alpha=0.8)
        plt.tight_layout()
        plt.show()
 
    def Ellipses(self):

        def build_ellipse(ipt, sx, sy, m_nu, name, vp):
            z = ipt.Z2(sx, sy, m_nu) 
            if abs(z.imag) > 0 or z.real < 0: return []
            z = z**0.5
            sx = sx.real
            sy = sy.real
            z  = z.real

            h = H_tilde(sx, sy, z, self.data, ipt.xkp).real

            h = h.dot(h.T)
            L = np.array([[1, -self.G2.delta.p, 0], [1, -self.G2.delta.m, 0], [0, 0, 1]])
            h = np.linalg.inv(L).T.dot(h).dot(np.linalg.inv(L))

            a, b, c = h[0][0], h[0][1], h[1][1]
            l1 = (a + c)/2 + np.sqrt( ( (a - c) * 0.5) ** 2 + b ** 2 )
            l2 = (a + c)/2 - np.sqrt( ( (a - c) * 0.5) ** 2 + b ** 2 )
            v1 = np.array([b, l1-a, 0]).reshape((3, 1))
            v2 = np.array([b, l2-a, 0]).reshape((3, 1))
            v3 = np.array([0,    0, 1]).reshape((3, 1))
            v1 = v1 / (sum(v1 ** 2)**0.5)
            v2 = v2 / (sum(v2 ** 2)**0.5)

            rz = np.concatenate((v1, v2, v3), -1)

            h = rz.T.dot(h).dot(rz)
            h = np.diag(np.diag(h)) ** 0.5
            p = rz.T.dot(np.diag(h)).dot(rz)
            p = np.linalg.inv(L).dot(p).dot(np.linalg.inv(L).T)

            phi = math.atan2(p[2], p[1])
            tau = math.asinh(p[2]/(ipt.eps * m_nu * math.sin(phi)))

            Sx, Sy, Sz = ipt.NuVec(tau, phi, m_nu)
            h = H_tilde(Sx, Sy, Sz, self.data, ipt.xkp)
            p = np.array([Sx, Sy, Sz]).real
            if self.plot_ellipse: vp.add_ellipse(h, name, p)
            return [[h, p]]

        sx, sy = self.ref.Sx, self.ref.Sy
        data = self.ref.solution
        if self.plot_ellipse: vp = packet(data["H_T"], data["neutrino"])
        else: vp = None

        self.best = {}
        for i in self.D:
            L = self.D[i]
            b, sx1, sy1  = L.b1, L.sx1, L.sy1

            L.solutions += build_ellipse(self.Z2_pp, sx1, sy1, L.m_nu1, i + "++", vp)
            L.solutions += build_ellipse(self.Z2_pm, sx1, sy1, L.m_nu1, i + "+-", vp)
            L.solutions += build_ellipse(self.Z2_mp, sx1, sy1, L.m_nu1, i + "-+", vp)
            L.solutions += build_ellipse(self.Z2_mm, sx1, sy1, L.m_nu1, i + "--", vp)
            
            L.solutions += build_ellipse(self.Z2_pp, b, 0, L.m_nu1, i + "x++", vp)
            L.solutions += build_ellipse(self.Z2_pm, b, 0, L.m_nu1, i + "x+-", vp)
            L.solutions += build_ellipse(self.Z2_mp, b, 0, L.m_nu1, i + "x-+", vp)
            L.solutions += build_ellipse(self.Z2_mm, b, 0, L.m_nu1, i + "x--", vp)

            b, sx2, sy2  = L.b2, L.sx2, L.sy2
            L.solutions += build_ellipse(self.Z2_pp, sx2, sy2, L.m_nu2, i + "++", vp)
            L.solutions += build_ellipse(self.Z2_pm, sx2, sy2, L.m_nu2, i + "+-", vp)
            L.solutions += build_ellipse(self.Z2_mp, sx2, sy2, L.m_nu2, i + "-+", vp)
            L.solutions += build_ellipse(self.Z2_mm, sx2, sy2, L.m_nu2, i + "--", vp)

            L.solutions += build_ellipse(self.Z2_pp, b, 0, L.m_nu2, i + "x++", vp)
            L.solutions += build_ellipse(self.Z2_pm, b, 0, L.m_nu2, i + "x+-", vp)
            L.solutions += build_ellipse(self.Z2_mp, b, 0, L.m_nu2, i + "x-+", vp)
            L.solutions += build_ellipse(self.Z2_mm, b, 0, L.m_nu2, i + "x--", vp)

        for k in range(len(L.solutions)):
            L.solutions[k].append(self.R_T.dot(L.solutions[k][-1]))
            v = self.R_T.dot(L.solutions[k][-1])
            x = sum((v - np.array([self.nu.px, self.nu.py, self.nu.pz]))**2) * (0.001**2)
            self.best[x.item()] = L.solutions[k]
        s = sorted(self.best)[0]
        print("__________")
        if self.plot_ellipse: vp.compile2D_Proj()
        print(s)
        exit()

    def debug(self):
        def ellipse_build(z2p, lp, m_nu, key, dic):
            sz = z2p.Z2(lp.sx1, lp.sy1, m_nu)
            dSx, dSy = lp.sx1 - z2p.Sx0, lp.sy1 - z2p.Sy0
            dSz = ((sz - z2p.Z2(z2p.Sx0, z2p.Sy0, m_nu)) ** 0.5).real
            phi = math.atan2(dSz * z2p.eps, -z2p.kappa.sin * dSx + z2p.kappa.cos * dSy)
            sh = (- z2p.kappa.sin * dSx + z2p.kappa.cos * dSy) / (m_nu * math.cos(phi))
            tau = math.asinh(sh.real)
#            if abs(tau) > 12: return 
            sx, sy, sz = z2p.NuVec(tau, phi, m_nu)
            kx = str(phi)[:12] + ":" + str(tau)[:12]
            if kx in dic: return
            h = H2_tilde(sx, sy, sz ** 2, self.data, z2p.xkp).real
            a, b, d = h[0][0], h[0][1], h[1][1]
            t = math.atan2(a - d, 2 * b) * 0.5
            hx = np.array([
                [math.cos(t), -math.sin(t), 0], 
                [math.sin(t),  math.cos(t), 0], 
                [  0,   0, 1]
            ])
            dic[kx] = [key, phi, tau, hx.T.dot(h).dot(hx)]

        sx, sy = self.ref.Sx, self.ref.Sy
        data = self.ref.solution
        vp = packet(data["H_T"], data["neutrino"]) if self.plot_ellipse else None
        kex = {}
        for i in self.D:
            L = self.D[i]
            ellipse_build(self.Z2_pp, L, L.m_nu1, i + "| ++ (nu1)", kex)
            ellipse_build(self.Z2_mp, L, L.m_nu1, i + "| -+ (nu1)", kex)
            ellipse_build(self.Z2_pm, L, L.m_nu1, i + "| +- (nu1)", kex)
            ellipse_build(self.Z2_mm, L, L.m_nu1, i + "| -- (nu1)", kex)

            ellipse_build(self.Z2_pp, L, L.m_nu2, i + "| ++ (nu2)", kex)
            ellipse_build(self.Z2_mp, L, L.m_nu2, i + "| -+ (nu2)", kex)
            ellipse_build(self.Z2_pm, L, L.m_nu2, i + "| +- (nu2)", kex)
            ellipse_build(self.Z2_mm, L, L.m_nu2, i + "| -- (nu2)", kex)
        for i in kex: vp.add_ellipse(kex[i][-1], kex[i][0], kex[i][-1].dot([[1], [1], [1]]))
        vp.compile2D_Proj()
        exit()


    def margins(self, sx, sy, x_min, y_min, x_max, y_max):
        sx, sy = np.array(sx), np.array(sy)
        mask = (sx >= x_min) * (sx <= x_max) * (sy >= y_min) * (abs(sy) <= y_max)
        return sx[mask], sy[mask]
 
    @property 
    def R_T(self):
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













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

        self.plot_lines()


    def plot_lines(self):
        ref = self.ref.solution
        vp = packet(ref["H_T"], ref["neutrino"], 1000)

        def pxpp(tau): return self.G2._Line(tau, +1, +1, +1)
        def pxmp(tau): return self.G2._Line(tau, +1, -1, +1)
        def pxpm(tau): return self.G2._Line(tau, +1, +1, -1)
        def pxmm(tau): return self.G2._Line(tau, +1, -1, -1)

        def mxpp(tau): return self.G2._Line(tau, -1, +1, +1)
        def mxmp(tau): return self.G2._Line(tau, -1, -1, +1)
        def mxpm(tau): return self.G2._Line(tau, -1, +1, -1)
        def mxmm(tau): return self.G2._Line(tau, -1, -1, -1)
        def get_pt(matrix): return np.array([matrix[0][2], matrix[1][2], matrix[2][1]]).real
        
        tau = 0 
        m = [
            get_pt(pxpp(tau)), get_pt(pxmp(tau)), 
            get_pt(pxpm(tau)), get_pt(pxmm(tau)), 
            get_pt(mxpp(tau)), get_pt(mxmp(tau)), 
            get_pt(mxpm(tau)), get_pt(mxmm(tau))
        ]

        Lins = {} 
        for i in range(len(m)):
            for j in range(len(m)):
                k = str(i) +"-"+ str(j)
                Lins[k] = linear_t(m[i], m[j])

        dy, dz = Lins["0-1"].d[1], Lins["0-1"].d[2]
        tau    = -0.5 * (np.atanh(dz / dy) if abs(dz) < abs(dy) else np.atanh(dy / dz) )
        Rx_mat = hyper_t(tau).Rx
        for k in Lins:
            Lins[k].d  = Rx_mat.dot(Lins[k].d)
            Lins[k].r0 = Rx_mat.dot(Lins[k].r0)

        u, v = Lins["0-1"].d, Lins["0-2"].d
        d_uv = u[0] * v[0] + u[1] * v[1]
        x_uv = u[0] * v[1] + u[1] * v[0]
        r = d_uv / x_uv

        tau = -0.5 * np.atanh(1 / r)
        Rz_hyper = hyper_t(tau).Rz
        for k in Lins:
            Lins[k].d  = Rz_hyper.dot(Lins[k].d)
            Lins[k].r0 = Rz_hyper.dot(Lins[k].r0)

        u   = Lins["0-1"].d
        tau = -np.asinh((u[1] / u[2]))
        Rz  = hyper_t(tau).Ry
        for k in Lins:
            Lins[k].d  = Rz.dot(Lins[k].d)
            Lins[k].r0 = Rz.dot(Lins[k].r0)

        #pitch_x = np.asinh(m[0][1] / m[0][2])
        #Rx_tilt = hyper_t(pitch_x).Rz
        #for k in Lins:
        #    Lins[k].d  = Rx_tilt.dot(Lins[k].d)
        #    Lins[k].r0 = Rx_tilt.dot(Lins[k].r0)

        #pitch_y =  np.acosh(m[0][0] / m[0][1]) 
        #Ry_tilt = hyper_t(pitch_y).Rz
        #for k in Lins:
        #    Lins[k].d  = Ry_tilt.dot(Lins[k].d)
        #    Lins[k].r0 = Ry_tilt.dot(Lins[k].r0)
 
        for k in Lins:
            #Lins[k].r0 += center
            vp.add_line(Lins[k], k)
        vp.compile2D_Proj()

    @property 
    def R_T(self):
        px, py, pz = self.lep.px, self.lep.py, self.lep.pz
        phi   = np.arctan2(py, px)
        theta = np.arctan2(np.sqrt(px**2 + py**2), pz)
        R_z   = angular_t(-phi).Rz
        R_y   = angular_t(0.5*np.pi - theta).Ry
        
        b_vec = np.array([self.jet.px, self.jet.py, self.jet.pz])
        b_rot = R_y @ (R_z @ b_vec)
        R_x = angular_t(-np.arctan2(b_rot[2], b_rot[1])).Rx
        self.RT = R_z.T @ R_y.T @ R_x.T
        return self.RT













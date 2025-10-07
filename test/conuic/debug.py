from visualize import *
from atomics import *
from shapes import *
from poly import *
import numpy as np

def _print(tlt, val = None, obj = None):
    if val is None: return print("====== " + tlt + " ======")
    if isinstance(val, list) and obj is not None: 
        o = ""
        for i in val: o += string(obj, i) + " "
        val = o
    if isinstance(val, dict):
        o = ""
        for i in val: o += string_(i, val[i]) + " "
        val = o

    print("-------- " + tlt + " ------")
    print(val)

class debug:
    def __init__(self):
        self.debug_mode = False

    def base_debug(self, idx):
        cx = self.engine[idx]
        _print("--------- pair: " + str(idx) + " ---------")
        if len(cx.truth_pair): print(">>> TRUTH PAIR <<<")
        _print("RT", cx.RT)
        _print("Z^2 Polynomial", ["A", "B", "C", "D", "E"], cx)
        _print("psi-angles", ["cpsi", "spsi", "tpsi"], cx)
        _print("Sx coefficients", ["a_x", "b_x", "c_x"], cx)
        _print("Sy coefficients", ["a_y", "b_y", "c_y"], cx)

        _print("H-Tilde Matrix")
        _print("Htc", cx.htc)
        _print("Ht1", cx.ht1)
        _print("Ht2", cx.ht2)

        _print("H Matrix")
        _print("Hc", cx.hc)
        _print("H1", cx.h1)
        _print("H2", cx.h2)

    def eigen_debug(self, idx):
        cx = self.engine[idx]
        _l, _z, _t = cx.l, cx.z, cx.tau

        # -------- P(L, Z, tau) ------- #
        P = cx.P(_l, _z, _t)
        #_print("P(lambda, Z, tau)"   , cx.P(   _l, _z, _t))
        #_print("dPdL(lambda, Z, tau)", cx.dPdl(_l, _z, _t))
        #_print("dPdZ(lambda, Z, tau)", cx.dPdz(_l, _z, _t))
        #_print("dPdt(lambda, Z, tau)", cx.dPdt(_l, _z, _t))

        #test = {"pred" : cx._transfer(_l, _z, _t), "truth" : P}
        #_print("transfer function P(lambda) = 1 / 3 (lambda dP/dL + Z dP/dZ)", test)
        print(">_________<")
        if len(cx.truth_pair): print(">>> TRUTH PAIR <<<")
        x = cx._roots_tau(_t)
        x = {
                "real"   : round(x["real"],3), "imag" : round(x["imag"], 3), 
                "P" : round(x["P"], 2), "dPdTau" : x["dPdtau"],  "tau" : x["tau"]
        }
        #_print("special root", x)
        cx._mobius()



        # -------- dP/dZ = 0 ------ #
        #r = cx.dPdz_l(_z, _t)
        #v = {i : cx.dPdz(l, _z, _t) for i, l in r.items()}
        #_print("dPdZ = 0", r)
        #_print("test dPdZ = 0 @ l0, l1", v)

        #print(cx._test(_l, _z, _t)) 



        #print(cx._dPdZ_D(_z, _t))






    def debug(self, idx):
        #self.base_debug(idx)
        self.eigen_debug(idx)






class traject:
    def __init__(self, runtime):
        self.runtime = runtime
        self.plots = []

    def shapes(self):
        htilde = self.figures(Ellipse, "b-", 1, True)
        self.ellipse(self.H_tilde, self.z, self.tau, htilde)
        htilde.alpha = 1.0

        pl_tilde = self.figures(Plane, "b", 1, False)
        self.plane(self.H_tilde, self.z, self.tau, pl_tilde)
        pl_tilde.alpha = 0.01

        ab = self.figures(Ellipsoid, "g", 1, False)
        ab.data.matrix = self.A_b(self.z, self.tau)
        ab.alpha = 0.1

        amu = self.figures(Ellipsoid, "r", 1, False)
        amu.data.matrix = self.A_mu(self.z, self.tau)
        amu.alpha = 0.1

        if not self.is_truth: return 
        htilde.linewidth = 2
        htilde.color = "r"

        pl_tilde.color = "r"
        pl_tilde.alpha = 0.01

    def figures(self, gen, c, lw, on = True):
        obj = gen(self, self.runtime.ax)
        obj.data = data()
        obj.linewidth = lw
        obj.color = c
        if not on: return obj
        self.plots.append(obj)
        return obj

    # ----- Ellipsoids of lepton and bquark
    def A_mu(self, z, t, nu = None): 
        if nu is None: nu = M_nu([3], [3], [self.m_nu**2])
        return p_Amu(self, z, t, nu)

    def A_b( self, z, t, nu = None): 
        if nu is None: nu = M_nu([3], [3], [self.m_nu**2])
        return  p_Ab(self, z, t, nu)

    # ----- plane of ellipse
    def plane(self, mtx, z, t, pl = None):   return p_plane(mtx(z, t), pl)
    def ellipse(self, mtx, z, t, pl = None): return p_ellipse(mtx(z, t), pl)







from reference import *
from visualize import *
from atomics import *
from shapes import *
from poly import *
import numpy as np

def _print(tlt, val = None, obj = None):
    if val is None: return print("====== " + tlt + " ======")
    if isinstance(val, list) and obj is not None: 
        o = ""
        for i in val: o += string(obj, i) + " \n"
        val = o
    if isinstance(val, dict):
        o = ""
        for i in val: o += string_(i, val[i]) + " \n"
        val = o

    print("-------- " + tlt + " ------")
    print(val)

def _assertions(tlt, trgt, val, limit =  0.01):
    try: assert (abs(trgt - val) / abs(trgt))*100 < limit
    except AssertionError: print(tlt + "->: ", trgt, val, "diff:", abs(trgt - val))

class debug:
    def __init__(self):
        self.debug_mode = False

    def reference_debug(self, cx):
        if not cx.is_truth: return 
        l, b = cx.lep.mass, cx.jet.mass

        p = Particle(0, 0, 0, 0)
        for i in cx.truth_pair:
            if cx.lep.hash == i.hash: continue
            if cx.jet.hash == i.hash: continue
            p = p + i
        nu = p
        t = p + cx.lep + cx.jet
        w = p + cx.lep
        ref = NuSol(cx.jet, cx.lep, w.mass, t.mass, cx.m_nu)
        rsx, rsy, rz2 = ref.Sx, ref.Sy, ref.Z2
        _assertions("Z2", rz2, cx.Z2(rsx, rsy)) 
        _assertions("mW", w.mass, cx.mW2(rsx)**0.5)
        _assertions("mT", t.mass, cx.mT2(rsx, rsy)**0.5)
 
        tau = cx.get_tau(rsx, rsy)
        csx, csy = cx.Sx(rz2**0.5, tau), cx.Sy(rz2**0.5, tau)
        _assertions("Sx", rsx, csx, 0.1)
        _assertions("Sy", csy, rsy, 0.1)

        _assertions("mW", cx.mW2(csx)**0.5     , w.mass)
        _assertions("mT", cx.mT2(csx, csy)**0.5, t.mass)

    def base_debug(self, cx):

        print("cos ", cx.cos )
        print("sin ", cx.sin )
        print("w   ", cx.w   )
        print("o   ", cx.o   )
        print("o2  ", cx.o2  )
        print("ml  ", cx.lep.mass)

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
        exit()



    def eigen_debug(self, cx):
        _l, _z, _t = cx.l, cx.z, cx.tau

        # -------- P(L, Z, tau) ------- #
        P = cx.P(_l, _z, _t)
        _print("P(lambda, Z, tau)"   , cx.P(   _l, _z, _t))
        _print("dPdL(lambda, Z, tau)", cx.dPdl(_l, _z, _t))
        _print("dPdZ(lambda, Z, tau)", cx.dPdz(_l, _z, _t))
        _print("dPdt(lambda, Z, tau)", cx.dPdt(_l, _z, _t))

        # -------- dP/dZ = 0 ------------ #
        _print("dP/dZ degeneracy")
        dg = cx._lambda_dPdZ_degenerate()
        for i in dg: _print("dPdZ - Degenerate roots", dg[i])

        # -------- dP/dL = 0 ------------ #
        _print("dP/dLambda degeneracy")
        dg = cx._lambda_dPdL_degenerate()
        for i in dg: _print("dPdL - Degenerate roots", dg[i])

        # ------- dP/dtau = 0 and P = 0 --------- #
        dg = cx._M_coef()
        _print("Mobius coefficients", dg)
        
        dg = cx._lambda_roots_dPdtau(_z, True)
        _print("dP/dtau = 0, P = 0", dg)

    def infer_debug(self, cx):
        p = Particle(0, 0, 0, 0)
        for i in cx.truth_pair:
            if cx.lep.hash == i.hash: continue
            if cx.jet.hash == i.hash: continue
            p = p + i
        nu = p
        t = p + cx.lep + cx.jet
        w = p + cx.lep
        print("==>", t.mass / 1000, w.mass / 1000)

    def debug(self, idx):
        cx = self.engine[idx]
        print("->", cx.lep)
        print("+>", cx.jet)
        _print("--------- pair: " + str(idx) + " ---------")
        if len(cx.truth_pair): print(">>> TRUTH PAIR <<<")
        #self.reference_debug(cx)
        #self.infer_debug(cx)
        self.base_debug(cx)
        #self.eigen_debug(cx)
        exit()


class traject:
    def __init__(self, runtime):
        self.runtime = runtime
        self.plots = []

    def shapes(self):
        l = 3 if len(self.parameters) > 2 else len(self.parameters)*0
        for i in range(l):
            self.z   = self.parameters[i].z
            self.tau = self.parameters[i].t
            if abs(self.tau) > 1: continue
            htilde = self.figures(Ellipse, "b-", 1, True)
            self.ellipse(self.H_matrix, self.z, self.tau, htilde)
            htilde.alpha = 1.0
            if not self.is_truth: continue
            htilde.linewidth = 2
            htilde.color = "r"
        
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
    def ellipse(self, mtx, z = None, t = None, pl = None): 
        if z is not None and t is not None: return p_ellipse(mtx(z, t), pl)
        return p_ellipse(mtx, pl)
    def line(self, r0, d, pl): 
        pl.data.r0 = r0
        pl.data.d  = d


from original import *
from shapes import *
from particle import *
from atomics import *
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
    try: 
        if isinstance(trgt, np.ndarray): assert sum(sum(abs(trgt - val) / abs(trgt + (trgt == 0))*100 > limit)) == 0
        else: assert (abs(trgt - val) / abs(trgt))*100 < limit
        print(tlt + "+>:\n", trgt, "\n", val, "diff:", abs(trgt - val))
        return True
    except AssertionError: print(tlt + "!>:\n", trgt, "\n", val, "diff:", abs(trgt - val))

class debug:
    def __init__(self):
        self.debug_mode = False

    def reference_debug(self):
        l, b = self.lep.mass, self.jet.mass
        p = Particle(0, 0, 0, 0)
        for i in self.truth_pair:
            if self.lep.hash == i.hash: continue
            if self.jet.hash == i.hash: continue
            p = p + i
        nu = p
        t = p + self.lep + self.jet
        w = p + self.lep
        ref = NuSol(self.jet, self.lep, w.mass, t.mass, self.m_nu)
        
        _assertions("cos"    , ref.c, self.cos)
        _assertions("sin"    , ref.s, self.sin)
        _assertions("omega"  , ref.w, self.w)
        _assertions("Omega^2", ref.Om2, self.o2)
        rsx, rsy, rz2 = ref.Sx, ref.Sy, ref.Z2
        cz, ct = self.GetTauZ(rsx, rsy)
        if cz is None: return 
        csx, csy = self.Sx(cz, ct), self.Sy(cz, ct)
        assert _assertions("Sx", rsx, csx, 0.1) == True
        assert _assertions("Sy", csy, rsy, 0.1) == True

        assert _assertions("y1", ref.y1, self.y1(cz, ct)) == True
        assert _assertions("x1", ref.x1, self.x1(cz, ct)) == True
        _assertions("mW", self.mW2(cz, ct)**0.5, w.mass) == True
        _assertions("mT", self.mT2(cz, ct)**0.5, t.mass) == True
        _assertions("Z2", rz2, self.Z2(csx, csy)) 

        _print("H-Tilde Matrix")
        assert _assertions("DIFF", ref.H_tilde, self.Htilde(cz, ct), 0.01)
        assert _assertions("HROT", ref.H, self.Hmatrix(cz, ct), 0.01)
        assert _assertions("RT", ref.R_T, self.RT, 0.00001)


    def special_debug(self):
        #print("fx:", self.mob.fixed_points())
        #print(self.mob.matrix())
        #print(self.mob.eigenval())
        #print(self.mob.eigenvec())

        r = self.mob.newton_method()
        if np.isnan(r): return
        md = self.mob.midpoint()
        print(r)
        print("->", self.mob.dPl0(md), self.mob.kfactor())
        print("->", self.mob.dPl0(r))
        self.error = self.mob.dPl0(r)
        self.tau   = math.atanh(r)
        print("+>", self.tau, math.atanh(md), math.atanh(self.mob.eigenval()[1].real))
        exit()

        lx = self.dPdtL0(1, self.tau)
        dp = self.dPdt(1, lx, self.tau)
        p  = self.P(1, lx, self.tau)
        self.lstar = lx

        vs, vsp = self.eigenvectors()
        #print("->", dp, p, lx)
        
        #tx = np.linspace(-2, 2, 10000)
        #import matplotlib.pyplot as plt
        #v1, v2 = self.mob.fixed_points()
        #plt.scatter(math.tanh(self.tau), self.mob.dPl0(self.tau, True), c = "blue", marker = "x")
        #plt.scatter(v1.real, self.mob.dPl0(v1.real), c = "red", marker = "*")
        #plt.scatter(v2.real, self.mob.dPl0(v2.real), c = "red", marker = "*")
        #plt.scatter(md, self.mob.dPl0(md), c = "black", marker = "o")
        #plt.plot(np.tanh(tx), self.mob.dPl0(tx, True))
        #plt.show()

        #p = Particle(0, 0, 0, 0)
        #for i in self.truth_pair:
        #    if self.lep.hash == i.hash: continue
        #    if self.jet.hash == i.hash: continue
        #    p = p + i
        #nu = p
        #    
        #from scipy.optimize import leastsq
        #es = [hx]
        #nx = np.array([nu.px, nu.py, nu.pz])
        #def nus(ts): return tuple(e.dot([math.cos(t), math.sin(t), 1]) for e, t in zip(es, ts))
        #def residuals(params): return sum(nus(params), -nx)
        #ts, lx = leastsq(residuals, [0], ftol=1e-12, epsfcn=0.0001)
        #print(nus(ts), nx, lx, ts)


        exit()
        print("vecs", vs, vsp)
        sx, sy = self.Sx(1000, self.tau), self.Sy(1, self.tau)
        print(sx, sy)

        for i in range(10000):
            mw, mt = self.masses(1000 + 10 * (i+1), self.tau)
            print(abs(mw - 81.325*1000), abs(mt - 172.1 * 1000))


        self.ellipse = Ellipse(self.ax)
        self.ellipse.color = "r" if self.is_truth else "k-"
        self.ellipse.data = data()
        self.ellipse.data.matrix = self.N() #self.Hmatrix(1000000, self.tau)
        self.ellipse.eign = vsp * 1000
        self.ellipse.theta = self.theta_star


    def debug(self):
        #print("->", self.lep)
        #print("+>", self.jet)
        _print("--------- pair: " + self.lep.hash + "-" + self.jet.hash + " ---------")
        if self.is_truth: print(">>> TRUTH PAIR <<<")
        else: print("__________ BACKGROUND ____________"); return
        #self.reference_debug()
        self.name = self.lep.hash + "-" + self.jet.hash
        self.special_debug()




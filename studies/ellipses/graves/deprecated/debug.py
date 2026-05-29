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

def _assertions(tlt, trgt, val, limit =  0.1):
    try: 
        if isinstance(trgt, np.ndarray): 
            assert sum(sum(abs(trgt - val) / abs(trgt + (trgt == 0))*100 > limit)) == 0
            print(tlt + "+>: diff:\n", abs(trgt - val))
            return True
        else: assert (abs(trgt - val) / abs(trgt))*100 < limit
        print(tlt + "+>:\n", trgt, val, "diff:", abs(trgt - val))
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
            break
        nu = p
        t = p + self.lep + self.jet
        w = p + self.lep

        ref = NuSol(self.jet, self.lep, w.mass, t.mass, self.m_nu)
        tau_Z = self.get_tauZ(ref.Sx, ref.Sy)
        ct, st = cosh(tau_Z.b), sinh(tau_Z.b)

        #assert _assertions("Z" , ref.Z, tau_Z.c)
        #assert _assertions("Sx", ref.Sx, self.SxP(st, ct, tau_Z.c) if ref.Sx > 0 else self.SxM(st, ct, tau_Z.c))
        #assert _assertions("Sy", ref.Sy, self.SyP(st, ct, tau_Z.c) if ref.Sx > 0 else self.SyM(st, ct, tau_Z.c))

        #assert _assertions("x1"    , ref.x1, self.x1(st, ct, tau_Z.c, tau_Z.d))
        #assert _assertions("y1"    , ref.y1, self.y1(st, ct, tau_Z.c, tau_Z.d))
        #assert _assertions("HTilde", ref.H_tilde, self.Htilde(st, ct, tau_Z.c, tau_Z.d))
        #assert _assertions("H"     , ref.H, self.H(st, ct, tau_Z.c, tau_Z.d))
        print("__________")
        print(tau_Z)
        exit()
        print(self.branch.eig)
        
        for i in self.branch.Zp() + self.branch.Zm():
            print(i)
            continue
            if abs(i[2].imag): continue

            try: tau_Z = self.get_tauZ(i[0].real, i[1].real)
            except: continue
            if abs(tau_Z.a.imag) > 0: continue
#            print(tau_Z.b)
            ct, st = cosh(tau_Z.b), sinh(tau_Z.b)
            print(tau_Z)
            #print("->", i[0], i[1], i[2])
            m = self.masses(st, ct, i[2].real) #tau_Z.c)
            l0 = self.dPl0(1, ct, st)
            k  = self.P(l0, 1, ct, st, tau_Z.d)
            o  = self.PL0(ct, st, tau_Z.d)
            #print(m)
            #if abs(o) > 1: continue
            print("->", l0, k, o, i) 
        exit()
        return 

        print(mt, mw)

#        assert _assertions("Z" , ref.Z, tau_Z.c)
#        assert _assertions("Sx", ref.Sx, self.SxP(st, ct, tau_Z.c) if ref.Sx > 0 else self.SxM(st, ct, tau_Z.c))
#        assert _assertions("Sy", ref.Sy, self.SyP(st, ct, tau_Z.c) if ref.Sx > 0 else self.SyM(st, ct, tau_Z.c))
#
#        assert _assertions("x1"    , ref.x1, self.x1(st, ct, tau_Z.c, tau_Z.d))
#        assert _assertions("y1"    , ref.y1, self.y1(st, ct, tau_Z.c, tau_Z.d))
#        assert _assertions("HTilde", ref.H_tilde, self.Htilde(st, ct, tau_Z.c, tau_Z.d))
#        assert _assertions("H"     , ref.H, self.H(st, ct, tau_Z.c, tau_Z.d))

#        H_  = self.H(st, ct, tau_Z.c, tau_Z.d)
        HT_ = self.Htilde(st, ct, tau_Z.c, tau_Z.d)
        print(self.lep.b, self.tpsi, self.o)
        vl, ve = np.linalg.eig(HT_)

 
        th = np.linspace(-0.9999, 0.9999, 1000000)
        ct, st = cosh(th), sinh(th)
        l0 = self.dPl0(tau_Z.c, ct, st)
        k  = self.P(l0, tau_Z.c, ct, st, tau_Z.d)
        o  = self.PL0(ct, st, tau_Z.d)
        
        idx = np.argmin(o)
        print(k[idx], o[idx], l0[idx])
        exit()

        agl = atan2(self.y1(st, ct, tau_Z.c, tau_Z.d), self.x1(st, ct, tau_Z.c, -tau_Z.d))

        nt  = np.array([nu.px, nu.py, nu.pz])
        jtv = np.array([self.jet.px, self.jet.py, self.jet.pz, self.jet.e])
        lpv = np.array([self.lep.px, self.lep.py, self.lep.pz, self.lep.e])

        th = np.linspace(0, 2 * np.pi, 1000000)
        nu  = self.H(st, ct, tau_Z.c, tau_Z.d).dot(np.array([np.cos(th), np.sin(th), np.ones_like(th)])).T

        ch = ( (nu - nt)**2).sum(-1)
        mn = np.argmin(ch)
        theta, ch = th[mn], ch[mn]

        dnu = self.H(st, ct, tau_Z.c, tau_Z.d).dot(np.array([-np.sin(th), np.cos(th), np.zeros_like(th)])).T

        wb = lpv + np.concatenate((nu, np.sqrt((nu ** 2).sum(-1)).reshape((-1, 1))), -1)
        tp = wb + jtv

        w = np.sqrt(wb[:, 3] ** 2 - (wb[:,:3] ** 2).sum(-1))
        t = np.sqrt(tp[:, 3] ** 2 - (tp[:,:3] ** 2).sum(-1))
        for i in range(1000000):
            print((nu - dnu)[i], th[i])
            if mn + 10 > i: continue
            exit()


        exit()

        print("________")
        print("ideal -> ", ch, theta)
        print(atan2(self.y1(st, ct, tau_Z.c, tau_Z.d), self.x1(st, ct, tau_Z.c,  tau_Z.d)))
        print(atan2(self.y1(st, ct, tau_Z.c, tau_Z.d), self.x1(st, ct, tau_Z.c, -tau_Z.d)))
        print(atan2(self.y1(st, ct, tau_Z.c, -tau_Z.d), self.x1(st, ct, tau_Z.c, tau_Z.d)))
        print(atan2(self.y1(st, ct, tau_Z.c, -tau_Z.d), self.x1(st, ct, tau_Z.c, -tau_Z.d)))




        nu_a   = self.H(st, ct, tau_Z.c, tau_Z.d).dot(np.array([np.cos(agl), np.sin(agl), 1])).T
        #print(ch, ((nu_a - nt)**2).sum(-1))
        #exit()

        #print(agl)
        #print(theta)
        #exit()















    def debug(self):
        print("->", self.lep)
        print("+>", self.jet)
        _print("--------- pair: " + self.lep.hash + "-" + self.jet.hash + " ---------")
        if self.is_truth: print(">>> TRUTH PAIR <<<")
        else: print("__________ BACKGROUND ____________"); return
        self.name = self.lep.hash + "-" + self.jet.hash
        try: return self.reference_debug()
        except AssertionError: 
            print("lepton", self.lep)
            print("jet", self.jet)

            l, b = self.lep.mass, self.jet.mass
            p = Particle(0, 0, 0, 0)
            for i in self.truth_pair:
                if self.lep.hash == i.hash: continue
                if self.jet.hash == i.hash: continue
                p = p + i
            nu = p
            t = p + self.lep + self.jet
            w = p + self.lep
            print("Masses: ", "top", t.mass, "w-boson", w.mass)







            exit()




from original import *
from particle import *
from atomics import *
from conix import *
import numpy as np


np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf) 

import math

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
 
        class tmp: 
            def __init__(self):
                self.mass = 0
        x = tmp()
        x.mass = 1000000000000000000000
        ref = NuSol(self.jet, self.lep, w.mass, t.mass, nu.mass, nu)
        print(w.mass, t.mass, nu.mass, ref.Sx, ref.Sy)
        nux = NuConuix(self.lep, self.jet, nu, ref)

        return 
        exit()

        rp = ref.Sx / ref.Sy
        cp = ref.x1 / ref.y1
        mtp = mT(nu.mass, self.lep, self.jet, ref.s, ref.c, ref.Sx, ref.Sy) 
#        print(nux.data._dG2.dp, nux.data._dG2.dm)
        print([math.acos(i) for i in [costheta(t, w), costheta(t, self.jet), costheta(w, nu), costheta(w, self.lep)]])


#        print(t.b, w.b, self.lep.b) 
        ref.use_minus = True
        rm = ref.Sx / ref.Sy
        cm = ref.x1 / ref.y1
        mtm = mT(nu.mass, self.lep, self.jet, ref.s, ref.c, ref.Sx, ref.Sy) 

        print(rp, rm, cp, cm, mtm, mtp)

        #for i in range(len(o2)):
        #    print(o2[i])
        #    x  = -2 * ref.Sx * (self.lep.p + self.jet.p * o2[i]) - 2 * self.jet.p * ref.Sy * o2[i] - self.lep.mass ** 2 + self.jet.mass ** 2 + nu.mass ** 2
        #    print(abs(x)**0.5, t.mass + w.mass + nu.mass + self.lep.mass + self.jet.mass, t.mass, w.mass) 

        cbW = costheta(w, self.jet)
        sbW = (1 - cbW**2)**0.5
        cWl = costheta(w, self.lep)
        sWl = (1 - cWl**2)**0.5
        
        clb = costheta(self.jet, self.lep)
        slb = (1 - clb**2)**0.5
        
        cphi = math.acos((clb - cbW * cWl) / (sbW * sWl))
        print(cphi * 180 / np.pi)

        a = - self.lep.e ** 2 * self.jet.e * self.jet.b * cbW * cWl + self.jet.e * self.lep.e - self.jet.mass * self.lep.mass
        b =   self.jet.e * self.lep.e ** 2 * self.jet.b * sbW * sWl
        print(math.atan2(b, a) * 180 / np.pi) #a / b)
#        plot_ellipses(nux.Htil + [[ref.H_tilde, "red", "-"]])


        exit()


    def debug(self):
#        _print("--------- pair: " + self.lep.hash + "-" + self.jet.hash + " ---------")
        #o2 = np.array(Omega0(self.jet, self.lep)).reshape(2, 2)
        if self.is_truth: print(">>> TRUTH PAIR <<<")
        else: print("__________ BACKGROUND ____________"); return
        self.name = self.lep.hash + "-" + self.jet.hash
        self.reference_debug()



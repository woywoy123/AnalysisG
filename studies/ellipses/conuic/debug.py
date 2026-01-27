from original import *
from particle import *
from atomics import *
from conix import *
from figures import *
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
        self.m_nu = nu.mass
        t = p + self.lep + self.jet
        w = p + self.lep
  
        nux = NuConuix(self.lep, self.jet, nu)
        ref = NuSol(self.jet, self.lep, w.mass, t.mass, nu.mass)
        print(ref.Sx, ref.Sy)
        _assertions("Z2", ref.Z2, nux.Z2(ref.Sx, ref.Sy, 1))
        _assertions("DeltaG - d1 x d2", - (1 - nux.pm.lb ** 2), (nux.pm.Dm * nux.pm.Dp).real)
        _assertions("Identity", np.diag([1, 1]), nux.pm.RV.T.dot(nux.pm.RV))
        _assertions("DeltaG - d1 + d2", - (nux.pm.Dm + nux.pm.Dp), (nux.pm.d1_p_d2).real)
  
        _assertions("x1", ref.x1, nux.x1(ref.Sx, ref.Sy, 1))
        _assertions("y1", ref.y1, nux.y1(ref.Sx, ref.Sy, 1))
        _assertions("w", ref.w      , nux._p.w)
        _assertions("o", ref.Om2**0.5, nux._p.o)

        _assertions("dG", nux.Z2(ref.Sx, ref.Sy, +1) - nux.Z2(ref.Sx, ref.Sy, -1), nux.DeltaG(ref.Sx, ref.Sy, 1))

        bs = nux.pm.SxSy(ref.Sx, ref.Sy)

        print("->", nux.Z2(ref.Sx, ref.Sy, 1))
        exit() 




        zp = nux._p
        zm = nux._m
        pm = nux.pm

        # Compute Sx, Sy from true neutrino momentum (if available)
        # Then compute Z^2 for both branches
        
        print(f"Using direct ω computation:")
        print(f"ω⁺ = {zp.w}")
        print(f"ω⁻ = {zm.w}")
        
        # Check that ω⁺ and ω⁻ satisfy the commutation relations
        print(f"\nCommutation relations:")
        print(f"ω⁺ + ω⁻ = {zp.w + zm.w} (expected: {-2*zp.cos/zp.sin})")
        print(f"ω⁺ - ω⁻ = {zp.w - zm.w} (expected: {2*zp.l_b/(zp.j_b*zp.sin)})")
        
        # Compute Z^2 at the true neutrino point (if available)
        if True:
            Z2_plus = Z2(zp, ref.Sx, ref.Sy)
            Z2_minus = Z2(zm, ref.Sx, ref.Sy)
            print(f"\nZ^2 at true neutrino point:")
            print(f"Z^2_+ = {Z2_plus}")
            print(f"Z^2_- = {Z2_minus}")
            print(f"Both should be near zero (within numerical precision).")
        
        # Test factorization of ΔG^2 at an arbitrary test point
        Sx_test, Sy_test = ref.Sx, ref.Sy  # or any other test point
        
        Z2_plus_test = Z2(zp, Sx_test, Sy_test)
        Z2_minus_test = Z2(zm, Sx_test, Sy_test)
        ΔG2_direct = Z2_plus_test - Z2_minus_test
        
        # Compute factored ΔG^2
        Γ_plus = (zp.w + zm.w) / (zp.o**2)
        Γ_minus = (zp.w - zm.w) / (zm.o**2)
        
        exit()
        from atomics import deltaR
        δ_plus = pm.Dp
        δ_minus = pm.Dm
        
        ΔG2_factored = -Γ_plus * Γ_minus * (Sx_test - δ_plus*Sy_test) * (Sx_test - δ_minus*Sy_test)
        
        print(f"\nFactorization test at (Sx, Sy) = ({Sx_test}, {Sy_test}):")
        print(f"ΔG² (direct)   = {ΔG2_direct}")
        print(f"ΔG² (factored) = {ΔG2_factored}")
        print(f"Difference     = {ΔG2_direct - ΔG2_factored}")
        









        exit()






    def debug(self):
        _print("--------- pair: " + self.lep.hash + "-" + self.jet.hash + " ---------")
        if self.is_truth: print(">>> TRUTH PAIR <<<")
        else: print("__________ BACKGROUND ____________"); return
        self.name = self.lep.hash + "-" + self.jet.hash
        self.reference_debug()













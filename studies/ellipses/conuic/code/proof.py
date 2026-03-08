from verified import *
from helper import *
import sympy as sp

def prove_masses():
    bq, mu, nu = particle("b"), particle("mu"), particle("nu")
    Sx, Sy, mT, mW, theta = symbols(["Sx", "Sy", "mt", "mW", "theta"])
    oj = verified(bq, mu, nu, theta)    

    x0  = - (1 / (2 * mu.E)) * (mW ** 2 - mu.mass ** 2 - nu.mass ** 2)
    sx_ =   (1 / mu.beta**2) * (x0 * mu.beta - mu.p * (1 - mu.beta**2)) - Sx
    sol = sp.solve(sx_, mW**2)[0]
    sol = mu.GetM2(mu.GetP(sol))

    clm = mu.GetM2(mu.GetP(oj.mW2(Sx)))
    assert sp.simplify(sol - clm) == 0

    x0p = - (1 / (2 * bq.E)) * (mT ** 2 - clm - bq.mass ** 2)
    sy_ = (1 / sp.sin(theta)) * (x0p / bq.beta - sp.cos(theta) * Sx) - Sy
    sol = sp.solve(sy_, mT**2)[0]

    clm = oj.mT2(Sx, Sy)
    clm = mu.GetP(mu.GetM2(clm))
    clm = bq.GetP(bq.GetM2(clm))

    sol = mu.GetP(mu.GetM2(sol))
    sol = bq.GetP(bq.GetM2(sol))
    assert sp.simplify(clm - sol).expand() == 0


def prove_Z2():
    w, Sx, Sy, mT, mW, theta = symbols(["w", "Sx", "Sy", "mt", "mW", "theta"])
    bq, mu, nu = particle("b"), particle("mu"), particle("nu")
    obj = verified(bq,mu, nu, theta)

    o2 = w ** 2 + 1 - mu.beta ** 2
    x0 = - (1 / (2 * mu.E)) * (obj.mW2(Sx) - mu.mass ** 2 - nu.mass ** 2)

    x1   = Sx - (Sx + w * Sy) / o2
    eps2 = (1 - mu.beta**2) * (obj.mW2(Sx) - nu.mass ** 2)
    z2   = x1 ** 2 * o2 - (Sy - w * Sx)**2 - (obj.mW2(Sx) - x0 ** 2 - eps2)

    Z2 = - ((o2 - 1) / o2) * Sx ** 2 + (2 * w / o2) * Sx * Sy  - ((1 - mu.beta**2)/o2) * Sy ** 2 
    Z2 = Z2 + (2 * mu.p * Sx) + mu.mass ** 2 - nu.mass ** 2

    dif = Z2.expand() - z2.expand()
    dif = mu.GetP(mu.GetM2(sp.simplify(dif))).expand()
    dif = sp.simplify(dif)
    assert dif == 0


def prove_dG2():
    wp, wm, Sx, Sy  = symbols(["w_{+}", "w_{-}", "Sx", "Sy"])
    obj = verified()
    G2P = obj.Z2(Sx, Sy, wp)
    G2M = obj.Z2(Sx, Sy, wm)

    dG2 = G2P - G2M

    tA = dG2.collect(Sx ** 2).coeff(Sx ** 2)
    tB = dG2.collect(Sx * Sy).coeff(Sx * Sy)
    tC = dG2.collect(Sy ** 2).coeff(Sy ** 2)
    
    clmA =   (wm - wp)*(wm + wp) / (obj.O2(wp)*obj.O2(wm))
    clmB = 2 * (wp * obj.O2(wm) - wm * obj.O2(wp)) / (obj.O2(wp)*obj.O2(wm))
    clmC = - (1 - obj.mu.beta**2) * clmA

    assert sp.simplify( (tA - clmA)) == 0
    assert sp.simplify( (tB - clmB)) == 0
    assert sp.simplify( (tC - clmC)) == 0

    GammaP = (wp + wm) / obj.O2(wp)
    GammaM = (wp - wm) / obj.O2(wm)
    Op, Om = obj.O(wp), obj.O(wm)
    dp = ((Op - Om) ** 2 - (wp + wm) ** 2) / (2 * (wp + wm))
    dm = ((Op + Om) ** 2 - (wp + wm) ** 2) / (2 * (wp + wm))
    
    # Vieta ---- sum of roots = -b / a, product of roots c/a
    assert sp.simplify( (dp + dm) - (-clmB) / clmA )  == 0
    assert sp.simplify( (dp * dm) - ( clmC) / clmA )  == 0

    fc = - GammaP * GammaM * (Sx - dp * Sy) * (Sx - dm * Sy)
    assert sp.simplify(dG2 - fc) == 0

    assert sp.simplify( (dp + dm) - 2 * (1 - obj.mu.beta**2 - wp * wm)/(wm + wp) ) == 0
    assert sp.simplify( (dp * dm) - (-(1 - obj.mu.beta**2)) ) == 0

def prove_dG2r():
    Z = verified()
    G = dG2()

    wp, wm = G.wpk, G.wmk
    Sx, Sy = symbols(["Sx", "Sy"])
    Sy_ = Sx * G.dp
    Sy_ = sp.together(sp.expand(Sy_, True))
    Sy_ = Sy_.subs(Sx, - Z.mu.mass ** 2 / (Z.mu.E * Z.mu.beta)) 
    Zp = Z.Z2(- Z.mu.mass ** 2 / (Z.mu.E * Z.mu.beta), Sy_, wp)

    x = sp.expand(Zp.subs(Z.mu.mass**2, (1 - Z.mu.beta**2) * Z.mu.E**2).subs(Z.bq.mass**2, (1 - Z.bq.beta**2) * Z.bq.E**2))
    sp.pprint(sp.solve(sp.simplify(sp.together(x))), Z.nu.mass)

    exit()
    Zm = Z.Z2(Sx, Sy_, wm)
    exit()

    sp.pprint(sp.simplify(sp.together(sp.expand(Zp + Zm))))
#    sp.pprint(Zm)


    exit()

    sp.pprint(sp.simplify(sp.solve(Zp, Sx)[0]))
    













import sympy as sp

def symbol(name): return sp.symbols(name, real = True, positive = True)
def symbols(lst): return [symbol(i) for i in lst]

class particle:
    def __init__(self, name):
        self.name   = name
        self.p      = symbol("p_"    + name)
        self.mass   = symbol("m_"    + name)
        self.beta   = symbol("beta_" + name)
        self.energy = symbol("E_"    + name)

def energy(m, p): return sp.sqrt(m ** 2 + p ** 2)
def mass(e, p): return sp.sqrt(e ** 2 - p ** 2)
def pmu(m, e):  return sp.sqrt(e ** 2 - m ** 2)

def beta(m, p): return sp.sqrt(1 - ( m / energy(m, p) ) ** 2)
def beta_mass_energy(prt): return beta(prt.mass, pmu(prt.mass, prt.energy))

def x1k(Sx, Sy, w, O): return Sx - (Sx - w * Sy) / O**2
def y1k(Sx, Sy, w, O): return Sy - (Sx - w * Sy)* w / O**2
def grad(y11, y22, x11, x22): return sp.simplify((y11 - y22)/(x11 - x22))

def roots(tpsi, tnh, pmu, m_mu, m_nu, s):
    a = pmu * tpsi 
    b = sp.sqrt(pmu**2 * tpsi ** 2 + tpsi * (tpsi + tnh) * (m_mu ** 2 - m_nu **2))
    return (a + s * b) / (tpsi + tnh)

def omega(mu, bq, s, theta): return (1 / sp.sin(theta)) * ( s * mu.beta / bq.beta - sp.cos(theta))

class pairs:
    def __init__(self, Sx, tpsi, name, s):
        self.Sx = sp.simplify(Sx)
        self.Sy = sp.simplify(Sx / tpsi)
        self.name = name
        self._Sx = symbol("Sx^" + s)
        self._Sy = symbol("Sy^" + s)


SxP, SyP, SxM, SyM, op, om, wm, wp = symbols(["Sx^+", "Sy^+", "Sx^-", "Sy^-", "O+", "O-", "w+", "w-"])
psip, psim, theta = symbols(["psi^+", "psi^-", "theta"])

mu = particle("mu")
nu = particle("nu")
bq = particle("bq")

tpsi_p, tpsi_m = sp.tan(psip), sp.tan(psim)

_op = sp.sqrt(wp ** 2 - sp.tan(psip) * sp.tan(psim))
_om = sp.sqrt(wm ** 2 - sp.tan(psip) * sp.tan(psim))


psx = {}
psx["Sp_Mp"]  = pairs((mu.p * tpsi_p) / (tpsi_p + sp.tan(theta)), tpsi_p, "S-max+", "+")
psx["Sp_rpp"] = pairs(roots(tpsi_p, sp.tan(theta), mu.p, mu.mass, nu.mass, +1), tpsi_p, "root ++", "+")
psx["Sp_rpm"] = pairs(roots(tpsi_p, sp.tan(theta), mu.p, mu.mass, nu.mass, -1), tpsi_p, "root +-", "+")
psx["Sp_rmp"] = pairs(roots(tpsi_m, sp.tan(theta), mu.p, mu.mass, nu.mass, +1), tpsi_m, "root -+", "+")
psx["Sp_rmm"] = pairs(roots(tpsi_m, sp.tan(theta), mu.p, mu.mass, nu.mass, -1), tpsi_m, "root --", "+")

msx = {}
msx["Sm_rpp"] = pairs(roots(tpsi_p, sp.tan(theta), mu.p, mu.mass, nu.mass, +1), tpsi_p, "root ++", "-")
msx["Sm_rpm"] = pairs(roots(tpsi_p, sp.tan(theta), mu.p, mu.mass, nu.mass, -1), tpsi_p, "root +-", "-")
msx["Sm_rmp"] = pairs(roots(tpsi_m, sp.tan(theta), mu.p, mu.mass, nu.mass, +1), tpsi_m, "root -+", "-")
msx["Sm_rmm"] = pairs(roots(tpsi_m, sp.tan(theta), mu.p, mu.mass, nu.mass, -1), tpsi_m, "root --", "-")
msx["Sm_Mm"] = pairs((mu.p * tpsi_m) / (tpsi_m + sp.tan(theta)), tpsi_m, "S-max-", "-")


_tpsi_p = (1 - mu.beta ** 2 - wp * wm + sp.sqrt( ((_op * _om)**2)))/ (wp + wm)
_tpsi_m = (1 - mu.beta ** 2 - wp * wm - sp.sqrt( ((_op * _om)**2)))/ (wp + wm)

x1pp = x1k(SxP, SyP, wp, _op)
y1pp = y1k(SxP, SyP, wp, _op)

x1pm = x1k(SxP, SyP, wm, _om)
y1pm = y1k(SxP, SyP, wm, _om)

x1mp = x1k(SxM, SyM, wp, _op)
y1mp = y1k(SxM, SyM, wp, _op)

x1mm = x1k(SxM, SyM, wm, _om)
y1mm = y1k(SxM, SyM, wm, _om)


lst = {}
lst["M_pppm"] = grad(y1pp, y1pm, x1pp, x1pm)
lst["M_mppp"] = grad(y1mp, y1pp, x1mp, x1pp)

lst["M_ppmp"] = grad(y1pp, y1mp, x1pp, x1mp)
lst["M_pmpp"] = grad(y1pm, y1pp, x1pm, x1pp)

lst["M_pmpm"] = grad(y1pm, y1pm, x1pm, x1pm)
lst["M_mpmp"] = grad(y1mp, y1mp, x1mp, x1mp)

lst["M_mppm"] = grad(y1mp, y1pm, x1mp, x1pm)
lst["M_pmmp"] = grad(y1pm, y1mp, x1pm, x1mp)

lst["M_pmmm"] = grad(y1pm, y1mm, x1pm, x1mm)
lst["M_mmmp"] = grad(y1mm, y1mp, x1mm, x1mp)

lst["M_mmpm"] = grad(y1mm, y1pm, x1mm, x1pm)
lst["M_mpmm"] = grad(y1mp, y1mm, x1pm, x1mm)

lst["M_mmpp"] = grad(y1mm, y1pp, x1mm, x1pp)
lst["M_ppmm"] = grad(y1pp, y1mm, x1pp, x1mm)

wp_ = omega(mu, bq, +1, theta)
wm_ = omega(mu, bq, -1, theta)

class expressions:

    def __init__(self, name, i, j):
        self.name = name
        self.y_x = i + " | "+ j
        self.SxSy = {}
        self.expression = None

unique = {}
for i in lst:
    for j in lst:
        if sp.simplify(lst[i] - lst[j]) == 0: continue
        eq = lst[i]
        sp.pprint(sp.simplify(lst[i]))
        for p in psx:
            for m in msx:
                exp = eq.subs(psx[p]._Sx, psx[p].Sx).subs(msx[m]._Sx, msx[m].Sx)
                exp = exp.subs(psx[p]._Sy, psx[p].Sy).subs(msx[m]._Sy, msx[m].Sy)
                print("----------------" + i + " -> " + j + " @ (" + p + ") -> (" + m + ") -----------------------")
                exp = sp.simplify(exp.subs(wp, wp_).subs(wm, wm_))

                id = str(hash(exp))
                if id in unique:
                    idx = str(hash(psx[p].Sx + msx[m].Sx))
                    unique[id].SxSy[idx] = [psx[p], msx[m]]
                    continue
                unique[id] = expressions(i + " -> " + j + " @ (" + p + ") -> (" + m + ")", i, j)
                idx = str(hash(psx[p].Sx + msx[m].Sx))
                unique[id].SxSy[idx] = [psx[p], msx[m]]
                unique[id].expression = exp
                sp.pprint(exp)







            
                










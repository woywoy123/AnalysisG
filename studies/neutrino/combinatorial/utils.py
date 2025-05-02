def makeDict(inpt):     return {i : 0  for i in inpt}
def percent(totl, rsd): return {i : 100*rsd[i] / totl[i] for i in totl}
def protect(inpt):  return None if not len(inpt) else inpt[0]
def get_nu(inpt):   return None if inpt is None else protect([i for i in inpt if i.is_nu ])
def get_jet(inpt):  return None if inpt is None else protect([i for i in inpt if i.is_b  ])
def get_ell(inpt):  return None if inpt is None else protect([i for i in inpt if i.is_lep])
def get_pxt(inpt): 
    x = [i for i in inpt if i is not None]
    if len(x) < 2: return None 
    return sum(x)


def routine(con, typx, param, fx1 = None, fx2 = None):
    for i in typx:
        if fx1 is not None: exp = con.get(i, False)
        else: exp = 0
        if exp != 0: fx1(exp, i, param)

        if fx2 is not None: exp = con.get(i, True)
        else: exp = 0
        if exp != 0: fx2(exp, i, param)

def make(data, fx1, fx2):
    out = []
    for i in data:
        for j in data[i]:
            for k in data[i][j]:
                rn = fx1(k)
                if rn is None: continue
                rn = fx2(rn)
                if rn is None: continue
                out.append(rn)
    return out


def make_atm(data, fx1, fx2 = None):
    out = []
    keys = list(data[0])
    for i in keys:
        for k in range(len(data[0][i])):
            if data[0][i][k] is None or data[1][i][k] is None: continue
            rn = fx1(data[0][i][k], data[1][i][k])
            if rn is None: continue
            if fx2 is None: out += rn; continue
            try: rn = fx2(rn)
            except TypeError: rn = fx2(rn, i)
            if rn is None: continue
            out += rn
    return out

def resize(data, slx):
    for i in list(data):
        if i in slx: continue
        try: data["other"] += data[i]
        except KeyError: data["other"] = data[i]
        del data[i]
    return data

def sumx(inpt, ox, fx, nfx = None):
    ix = range(len(inpt))
    if nfx is None: 
        for i in ix: ox += fx(inpt[i])
        return ox

    for i in ix: 
        if not nfx(inpt[i]): continue
        ox += fx(inpt[i])
    return ox

def bar(p, sym): return sym if p > 0 else "\\bar{" + sym + "}"

def safe_pair(inpt):
    n1, n2 = inpt
    if len(n1) and len(n2):     return n1  , n2
    if len(n1) and not len(n2): return None, n2
    if not len(n1) and len(n2): return n1, None
    return None, None

def pdgid(p):
    if abs(p) == 11: return bar(p, "e")
    if abs(p) == 12: return bar(p, "\\nu_{e}")
    if abs(p) == 13: return bar(p, "\\mu")
    if abs(p) == 14: return bar(p, "\\nu_{\\mu}")
    if abs(p) == 15: return bar(p, "\\tau")
    if abs(p) == 16: return bar(p, "\\nu_{\\tau}")
    return ""

def get_pdg(p):
    kx = ""
    dx = {}
    xt = [i for i in p if i is not None]
    if not len(xt): return "None"
    for i in xt:
        pdg = abs(i.pdgid)
        try: dx[pdg] += [i]
        except KeyError: dx[pdg] = [i]
    for i in sorted(list(dx)): kx += "".join([pdgid(j.pdgid) for j in dx[i]])
    return "(" + kx + ")"

def extract(dx):
    kx = ""
    for i, j in dx.items():
        if j is None: continue
        kx += j.symbolic
    return kx if len(kx) else None

def colors():
    return iter(sorted(list(set([
        "aqua", "orange", "green","blue","olive","teal","gold",
        "darkblue","lime","crimson","magenta","orchid",
        "sienna","salmon","chocolate", "navy", "plum", "indigo", 
    ]))))

def Fancy(tl, sub = None):
    if sub is None and tl == "top_children": return "(TopChildren)"
    if sub is None and tl == "truthjet"    : return "(TruthJets)"
    if sub is None and tl == "jetchildren" : return "(JetsChildren)"
    if sub is None and tl == "jetleptons"  : return "(JetsLeptons)"
    if sub is None and tl == "truth"       : return "Truth"
    if tl == "top_children": pr = "quark"
    else: pr = "jet"

    if sub == "sets"  : return "Correct ($\\ell$ and b-" + pr + ") Sets"
    if sub == "swp_b" : return "Swapped b-" + pr
    if sub == "swp_l" : return "Swapped $\\ell$"
    if sub == "swp_bl": return "Swapped $\\ell$ and b-" + pr
    if sub == "fake"  : return "Fake or Unmatched"
    return tl


def get_rtop(p): return None if p is None else p.recotop
def get_mass(p): return p.Mass / 1000.0
def get_swp_mass(data): return [sum(data[0]).Mass / 1000.0, sum(data[1]).Mass / 1000.0]

def get_deltaR(p, swp = False): return [p[0][0].DeltaR(p[swp][1]), p[1][0].DeltaR(p[not swp][1])]
def make_deltaR(p): return get_deltaR(p[0]) + get_deltaR(p[1])

def get_nus_b(p1, p2): 
    if p1.tru_b is None: return None
    if p1.rnu   is None: return None
    if p2.tru_b is None: return None
    if p2.rnu   is None: return None
    return [[p1.tru_b, p1.rnu], [p2.tru_b, p2.rnu]]

def get_nus_l(p1, p2): 
    if p1.tru_l is None: return None
    if p1.rnu   is None: return None
    if p2.tru_l is None: return None
    if p2.rnu   is None: return None
    return [[p1.tru_l, p1.rnu], [p2.tru_l, p2.rnu]]


def get_chi2_nus(data, sym):
    t1, t2 = data
    nu1 = [i for i in t1 if i.__class__.__name__ == "Neutrino"]
    nu2 = [i for i in t2 if i.__class__.__name__ == "Neutrino"]
    if not len(nu1) or not len(nu2): return None
    asym = abs(nu1[0].pt - nu2[0].pt) / 1000.0
    fxc = 1.0 / 1000000
    return [(
        nu2[0].chi2*fxc, nu1[0].chi2*fxc, asym, nu1[0].ellipse, 
        "$" + sym.replace(")(", ",").replace(")", "").replace("(","") + "$"
    )]

def fix_swapped(p1, p2): 
    if p1 is None and p2 is None: return None
    out = [[], []]
    if p1.rnu.matched_lepton ==  1 and p2.rnu.matched_lepton ==  1: out[0]+=[p1.tru_l]; out[1]+=[p2.tru_l]
    if p1.rnu.matched_lepton == -1 and p2.rnu.matched_lepton == -1: out[0]+=[p2.tru_l]; out[1]+=[p1.tru_l]
    if p1.rnu.matched_lepton ==  0 and p2.rnu.matched_lepton ==  0: out[0]+=[p1.tru_l]; out[1]+=[p2.tru_l]

    if p1.rnu.matched_bquark ==  1 and p2.rnu.matched_bquark ==  1: out[0]+=[p1.tru_b]; out[1]+=[p2.tru_b]
    if p1.rnu.matched_bquark == -1 and p2.rnu.matched_bquark == -1: out[0]+=[p2.tru_b]; out[1]+=[p1.tru_b]
    if p1.rnu.matched_bquark ==  0 and p2.rnu.matched_bquark ==  0: out[0]+=[p1.tru_b]; out[1]+=[p2.tru_b]
    out[0] += [p1.rnu]; out[1] += [p2.rnu]
    return out

def get_dr_nus(p1, p2):
    b = get_nus_b(p1, p2)
    l = get_nus_l(p1, p2)
    if b is None or l is None: return None
    out  = get_deltaR(b)
    out += get_deltaR(l)
    return out

def fix_dr_nus(p1, p2):
    b = get_nus_b(p1, p2)
    l = get_nus_l(p1, p2)
    if b is None or l is None: return None
    b1 = get_deltaR(b, False)
    b2 = get_deltaR(b, True)
    if b1 < b2: bc = b
    else: bc = [[b[1][0], b[0][1]], [b[0][0], b[1][1]]]

    l1 = get_deltaR(l, False)
    l2 = get_deltaR(l, True)
    if l1 < l2: lc = l
    else: lc = [[l[1][0], l[0][1]], [l[0][0], l[1][1]]]
    return [bc, lc]

def get_dr_mass(data):
    t1 = sum(data[0][0] + [data[1][0][0]]).Mass / 1000.0
    t2 = sum(data[0][1] + [data[1][1][0]]).Mass / 1000.0
    return [t1, t2]



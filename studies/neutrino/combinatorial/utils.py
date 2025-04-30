def makeDict(inpt):     return {i : 0  for i in inpt}
def percent(totl, rsd): return {i : 100*rsd[i] / totl[i] for i in totl}

def protect(inpt):      return None if not len(inpt) else inpt[0]
def get_nu(inpt):   return None if inpt is None else protect([i for i in inpt if i.is_nu ])
def get_jet(inpt):  return None if inpt is None else protect([i for i in inpt if i.is_b  ])
def get_ell(inpt):  return None if inpt is None else protect([i for i in inpt if i.is_lep])
def get_pxt(inpt): 
    x = [i for i in inpt if i is not None]
    if len(x) < 2: return None 
    return sum(x)

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
    print("!!!!!!!!!", p); exit()

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
        "darkblue","azure","lime","crimson","cyan","magenta","orchid",
        "sienna","silver","salmon","chocolate", "navy", "plum", "indigo", 
        "lightblue", "lavender", "ivory"
    ]))))

def Fancy(tl, sub = None):
    if sub is None and tl == "top_children": return "(TopChildren)"
    if sub is None and tl == "truthjet"    : return "(TruthJets)"
    if sub is None and tl == "jetchildren" : return "(JetsChildren)"
    if sub is None and tl == "jetleptons"  : return "(JetsLeptons)"
    if sub is None and tl == "truth"       : return "Truth"
    if tl == "top_children": pr = "quark"
    else: pr = "jet"

    if sub == "sets"  : return "Correct"
    if sub == "swp_b" : return "Swapped b-" + pr
    if sub == "swp_l" : return "Swapped lepton"
    if sub == "swp_bl": return "Swapped lepton and b-" + pr
    if sub == "fake"  : return "Fake or Unmatched"
    return tl






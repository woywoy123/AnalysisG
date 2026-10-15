import pickle
import pathlib
from plots import *

def depkl(pth): return pickle.load(open(pth, "rb"))
def svpkl(obj, pth): pickle.dump(obj, open(pth, "wb"))
def sms_lst(a): return sum(a, [])
def unq_lst(a): return list(set(a))

def to_str(inpt): 
    try: return inpt.decode("utf-8")
    except: return inpt

def mk_pth(inpt):
    try: pathlib.Path(inpt).mkdir(parents = True, exist_ok = True)
    except: return False
    return True

def pth_fm(bds, inpt): 
    return [ (bds + "/" + i.lstrip("/")).replace("//", "/") for i in inpt]

def get_dk(a, k = -1):
    try: return a[k] if k > -1 else a
    except KeyError: return []
    except IndexError: return []

def add_dk(a, k):
    try: a[k] 
    except KeyError: a[k] = {}
    except IndexError: return []
    return a

def add_ls(a, k):
    try:a[k]; return a
    except KeyError: a[k] = []
    return a

def mk_str(a, b, dl = " "): return str(a) + ": " + str(b) + dl

def fancy_title(flt, msk, tru):
    flt, msk, tru = bool(flt), bool(msk), bool(tru)

    # // --- Raw Truth -> uncut 
    # // --- Adj Truth -> cut filtered
    a1 = [["Unmasked", "Masked"][msk], ["Nominal" , "Truth" ][tru]]
    b1 = ["Raw","Adj"][flt]

    # // --- Nominal Raw Tops -> MLP(e, i, j) [truth uncut]
    # // --- Nominal Adj Tops -> MLP(e, i, j) [truth adjust cut]
    sy = b1 + "_" + a1[0] + "_" + a1[1] + "_Tops"

    # // --- Unmasked Raw Tops -> PR(M) [truth uncut]
    # // --- Unmasked Adj Tops -> PR(M) [truth cut]
    a0 = [["Unmasked", "Masked"][msk], ["Nominal" , "Truth" ][tru]]
    b0 = ["Raw","Adj"][flt]

    # // --- Masked Raw Tops -> PR(M) [truth uncut]
    # // --- Masked Adj Tops -> PR(M) [truth cut]
    sx = a0[1] + " [" + a0[0] + "] " + " " + b0 + ". Tops"
    sx = " ".join([k for k in sx.split(" ") if len(k)])
    return (sy, sx) 


def colors():
    return iter(list([
        "aqua", "orange", "green", "blue",
        "olive","teal","gold", "darkblue",
        "lime", "crimson","magenta","orchid", 
        "sienna","salmon","chocolate", "navy", 
        "plum", "indigo", "violet", "dodgerblue", 
        "slategray", "ivory"
    ]))

def sort_hist(hst):
    dcs = {}
    for i in hst:
        c = sum(i.counts)
        if c not in dcs: dcs[c] = []
        dcs[c] += [i]
    return sms_lst([dcs[i] for i in sorted(dcs)])


def matrixP(dt):
    def half(i, k = 0): return [" "] * int( 6 - (len(i) - k) / 2 )
    def kf(i):
        name = str(i)
        return "".join( half(name, len(name) % 2) + [name] + half(name)  )

    for i in range(len(dt) ** 2):
        try: dt[i % 5][i/5]
        except KeyError: dt[i % 5][i / 5] = 0
    
    fx  = ["|".join([">" + kf(i) + "<" for i in dt])]
    fx += ["=" * len(fx[-1])]
    fx += ["|".join([">" + kf(dt[i][j]) + "<" for i in dt ]) for j in dt ]
    fx += ["=" * len(fx[-1])]
    return "\n".join(fx)

def get_type(sym):
    am = {
        "$t\\bar{t}$": [
            "$t\\bar{t}$", 
            "$t\\bar{t} \\rightarrow \\ell$", 
            "$t\\bar{t} \\rightarrow \\ell \\bar{\\ell}$"
        ],
        "Single Top":  [
            "$tW$", 
            "$t_{\\text{s-channel}}$", 
            "$t_{\\text{t-channel}}$"
        ],
        "$t\\bar{t}V / t\\bar{t}H$": [
            "$t\\bar{t}W$", 
            "$t\\bar{t}H$", 
            "$t\\bar{t}Z\n \\rightarrow \\nu \\bar{\\nu}$", 
            "$t\\bar{t}Z \\rightarrow q \\bar{q} $"
        ],
        "V + Jets": [
            "$Z \\rightarrow \\ell \\bar{\\ell}$", 
            "$W \\rightarrow \\ell \\bar{\\nu}$"
        ],
        "Multiboson Rare": [
            "$WZ$", "$ZZ$", 
            "$WH$", "$ZH$", 
            "$\\ell\\ell\\nu\\nu$", 
            "$\\ell\\ell\\ell\\nu$", 
            "$\\ell\\ell\\ell\\ell$"
        ],
        "$t\\bar{t}t\\bar{t}$ (SM)": ["$t\\bar{t}t\\bar{t}$ (SM)"]
    }
    
    sig = {
        "$t\\bar{t}t\\bar{t}$ (400 GeV)":  "$t\\bar{t}t\\bar{t}$ (400)",
        "$t\\bar{t}t\\bar{t}$ (500 GeV)":  "$t\\bar{t}t\\bar{t}$ (500)", 
        "$t\\bar{t}t\\bar{t}$ (600 GeV)":  "$t\\bar{t}t\\bar{t}$ (600)", 
        "$t\\bar{t}t\\bar{t}$ (700 GeV)":  "$t\\bar{t}t\\bar{t}$ (700)",
        "$t\\bar{t}t\\bar{t}$ (800 GeV)":  "$t\\bar{t}t\\bar{t}$ (800)",
        "$t\\bar{t}t\\bar{t}$ (900 GeV)":  "$t\\bar{t}t\\bar{t}$ (900)",
        "$t\\bar{t}t\\bar{t}$ (1000 GeV)": "$t\\bar{t}t\\bar{t}$ (1000)"
    }
 
    def fx(name):
        t = name.replace(" ", "")
        t = t.replace("\\", "<x>")
        t = t.replace("\n", "")
        t = t.replace("$", "")
        t = t.replace("<x>", ".") 
        return t

    x = fx(sym)
    for k in am:
        for l in am[k]:
            n = fx(l)
            if n != x: continue
            return l, k

    for k in sig:
        l = sig[k]
        if fx(l) != x: continue
        return l, k

    return None, None


def as_str(dc, mx = 10):
    lx, ly = [str(i) for i in dc]    , [str(i) for i in dc.values()]
    lt, _  = [len(i) + 2 for i in ly], max([len(i) + 2 for i in ly])
    out = ""
    for i in range(len(lx)): 
        out += " " + lx[i] + " : "
        out += "".join([" "] * abs(mx - lt[i])) 
        out += ly[i] + "| "
    return "{" + out + "}"


def summary_stat(data):
    skx = {}
    for p_ in data:
        name, prc = p_.split("|")
        if name not in skx: skx[name] = []
        xk = prc + " : " + str(len(data[p_])) + ", "
        skx[name].append(xk)
    return skx


def summary_top(data, kfld):
    nrm = kfld.lower().strip("\n")
    skx = {}
    for px in data:
        name, prc = px.split("|") 
        if kfld not in name: continue 
        name, prc = get_type(prc)
        for i in range(len(data[px])):
            ntop = str(len(data[px][i]))
            if prc not in skx: skx[prc] = {}
            try:    skx[prc][ntop] += 1 
            except: skx[prc][ntop]  = 1
    return skx

def cuts(dt, fx = None):
    def kx(i): return True
    if fx is None: fx = kx
    return [i for i in dt if fx(i)] 

def average(lst):
    av = sum(lst) / float(len(lst))
    dv = sum([(av - i)**2 for i in lst]) / float(len(lst))
    return {"av" : av, "s2" : dv}

def get_mass(j):
    lx = {}
    for k in j:
        name, prc = get_type(k.prc)
        if prc not in lx: lx[prc] = {"m" : [], "w" : [], "hw" : []}
        lx[prc]["m"] += [i for i in k.masses]
        lx[prc]["w"] += [i for i in k.weight]

    for k in lx:
        lx[k]["hw"].append(basic_th1f(k, lx[k]["m"]))
        lx[k]["hw"].append(basic_th1f(k, lx[k]["w"]))
   
    mw, ms = [], [] 
    lx["prc"] = list(lx)
    for k in lx["prc"]: ms += lx[k]["m"]; mw += lx[k]["w"]
    lx["gm"] = ms; lx["gw"] = mw
    return lx

def make_sequence(msk, nr, name, rlx):
    hsk = get_mass(msk)
    km = "./Plots/" + name.replace(".", "/") + "/"

    _dx = "Invariant Mass of Tops (GeV)"
    _dy = "Number of Tops (Arb.) / (1 GeV)"
    hx = [hsk[f]["hw"][0] for f in hsk["prc"]]
    hx = [i for i in hx if i.Title.count("GeV")] + [i for i in hx if not i.Title.count("GeV")]
    hx = [i for i in hx if i.Title.count("$t\\bar{t}$") != 1] + [i for i in hx if i.Title.count("$t\\bar{t}$") == 1]

    th = templ_th1f("", 400, 0, 400, [], _dx, _dy, hx, colors())
    th.xStep    = 20
    th.Filename = nr.lower()
    th.Stacked  = True
    th.OutputDirectory = km + "plots"
    th.SaveFigure()

    _dx = "Top Candidate Score (Arb.)"
    _dy = "Number of Tops (Arb.)"
    hx = [hsk[f]["hw"][1] for f in hsk["prc"]]
    hx = [i for i in hx if i.Title.count("GeV")] + [i for i in hx if not i.Title.count("GeV")]
    hx = [i for i in hx if i.Title.count("$t\\bar{t}$") != 1] + [i for i in hx if i.Title.count("$t\\bar{t}$") == 1]

    th = templ_th1f("", 100, 0, rlx, [], _dx, _dy, hx, colors()) 
    th.xStep    = 1 if rlx > 1 else 0.2
    th.Stacked  = True
    th.Filename = nr.lower()
    th.OutputDirectory = km + "ranks" 
    th.SaveFigure()



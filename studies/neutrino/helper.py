from classes import atomic, Container
from tqdm import tqdm
from nusol import *
from multiprocessing import Process
import pathlib
import pickle
import gc

def build_samples(pth, pattern): return [str(i) for i in pathlib.Path(pth).glob(pattern) if str(i).endswith(".pkl")]

def chi2(t, r): return ((t.px - r.px)**2 + (t.py - r.py)**2 + (t.pz - r.pz)**2)*(1.0 / 1000.0)**2
def get_masses(nu, b, lep): return ((nu + b + lep).Mass, (nu + lep).Mass) if nu is not None else (-1, -1)
def get_static(ev, name): return get_best(ev.TruthNeutrinos, ev.StaticNeutrino[name])
def get_dynamic(ev, name): return get_best(ev.TruthNeutrinos, ev.DynamicNeutrino[name])
def get_pairs(ev, name, ix): 
    px = ev.Particles[name]
    if not len(px[0]): return None, None
    return [k for k in px[ix] if k.is_lep][0], [k for k in px[ix] if k.is_b][0]

def get_best(tnus, rnus):
    out = {}
    if not len(rnus[0]): return None, None
    for i in range(len(rnus[0])):
        chi2_1 = chi2(tnus[0], rnus[0][i])
        chi2_2 = chi2(tnus[1], rnus[1][i])
        rnus[0][i].chi2 = chi2_1
        rnus[1][i].chi2 = chi2_2
        out[chi2_1 + chi2_2] = [rnus[0][i], rnus[1][i]]
    return out[sorted(out)[0]]

def make_types(nus, leps, bs):
    try: return {i : [nus[i], leps[i], bs[i]] for i in range(2)}
    except IndexError: return {i : [] for i in range(2)}

def vector(inpt): return [i.vec for i in inpt]
def MPNuNu(k):
    from AnalysisG.core.tools import Tools
    data = []
    h = Tools().hash(str(k))
    k = pickle.loads(k)
    for d in range(len(k)):
        bs, ls, ev, mW1, mT1, mW2, mT2, n = k[d]
        data.append([n, DoubleNu(bs, ls, ev, mW1, mT1, mW2, mT2)])
    pickle.dump(data, open("./data/" + h + ".pkl", "wb"))

def compiler(lx, name, build = False):
    mW = 80.385*1000
    mT = 172.62*1000

    try: return pickle.load(open(name + ".pkl", "rb"))
    except: conx = [None for i in range(len(lx))]
    th = 120

    b = 0
    l = len(lx)
    runx = [None for _ in range(th)]
    prc = [[] for _ in range(th)]
    for ix in tqdm(range(l*build)):
        if conx[ix] is not None: continue
        i = lx[ix]
        if not len(i.TruthNeutrinos): continue
        nu1, nu2 = i.TruthNeutrinos

        # ----- input particles ----- #
        try:
            l1, b1 = get_pairs(i, name, 0)
            l2, b2 = get_pairs(i, name, 1)
            mt1, mw1 = get_masses(nu1, b1, l1)
            mt2, mw2 = get_masses(nu2, b2, l2)
        except: continue
        bsx, lsx = vector((b1, b2)), vector((l1, l2))

        # ----- Static Masses ------ #
        kx1 = (bsx, lsx, i.vec, mW , mT , mW , mT , name + "/stat/" + str(ix))
        # ----- Dynamic Masses ----- #
        kx2 = (bsx, lsx, i.vec, mw1, mt1, mw2, mt2, name + "/dyn/" + str(ix))
        prc[b] += [kx1, kx2]
        if len(prc[b]) < 1000: continue
        p1 = Process(target=MPNuNu, args=(pickle.dumps(prc[b]),))
        p1.start()
        runx[b] = p1
        prc[b] = []
        rn = True
        while rn:
            for b in range(len(runx)):
                if runx[b] is None: rn = False; break
                if runx[b].is_alive(): continue
                runx[b].join()
                rn = False
                runx[b] = None
                break

    for x in range(len(prc)):
        if not len(prc[x]): continue
        p1 = Process(target=MPNuNu, args=(pickle.dumps(prc[x]),))
        p1.start()
        runx[x] = p1
    for y in runx:
        if y is None: continue
        y.join()

    data = []
    nex = build_samples("./data", "*.pkl")
    for k in nex: data += pickle.load(open(k, "rb"))
    prc = [[None, None] for _ in range(l)]
    for i in data:
        nx, nus = i
        base, mode, idx = nx.split("/")
        if name != base: continue
        idx, idy = int(idx), (0 if mode == "stat" else 1)
        prc[idx][idy] = nus
    data = []

    update = False
    for ix in tqdm(range(l)):
        i = lx[ix]
        if conx[ix] is not None:      continue
        if not len(i.TruthNeutrinos): continue
        nu1, nu2 = i.TruthNeutrinos

        # ----- input particles ----- #
        try:
            l1, b1 = get_pairs(i, name, 0)
            l2, b2 = get_pairs(i, name, 1)
            mt1, mw1 = get_masses(nu1, b1, l1)
            mt2, mw2 = get_masses(nu2, b2, l2)
        except: continue
        bsx, lsx = vector((b1, b2)), vector((l1, l2))
        snu1_, snu2_ = get_best({0 : nu1, 1 : nu2}, prc[ix][0].nunu_s())
        dnu1_, dnu2_ = get_best({0 : nu1, 1 : nu2}, prc[ix][1].nunu_s())
        prc[ix] = None

        con = Container(
                make_types((nu1, nu2), (l1, l2), (b1, b2)), # truths
                make_types(get_dynamic(i, name), (l1, l2), (b1, b2)), # cuda_dyn
                make_types(get_static(i , name), (l1, l2), (b1, b2)), # cuda_static
                make_types((dnu1_, dnu2_), (l1, l2), (b1, b2)), # reference dynamic
                make_types((snu1_, snu2_), (l1, l2), (b1, b2))  # reference static
        )
        con.event_data["met"] = i.met
        con.event_data["phi"] = i.phi
        conx[ix] = con

        update = True
        prc[ix] = None
        if ix % 10000 == 9999: pickle.dump(conx, open(name + ".pkl", "wb"))
    if not update: return conx
    pickle.dump(conx, open(name + ".pkl", "wb"))
    return conx

def compxl(sl = None):
    return {
            "topchildren" : compiler(sl.Events if sl is not None else None, "top_children", True),
            "truthjet"    : compiler(sl.Events if sl is not None else None, "truthjet", True),
            "jetchildren" : compiler(sl.Events if sl is not None else None, "jetchildren", True),
            "jetleptons"  : compiler(sl.Events if sl is not None else None, "jetleptons", True)
    }


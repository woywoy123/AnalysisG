from classes import atomic, Container
from tqdm import tqdm
from nusol import *
import pickle

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

def compiler(lx, name):
    mW = 80.385*1000
    mT = 172.62*1000

    conx = [None for i in range(len(lx))]
    try: conx = pickle.load(open(name + ".pkl", "rb"))
    except: pass

    update = False
    for ix in tqdm(range(len(lx))):
        if conx[ix] is not None: continue

        i = lx[ix]
        if not len(i.TruthNeutrinos): continue
        nu1, nu2 = i.TruthNeutrinos

        # ----- input particles ----- #
        try: l1, b1 = get_pairs(i, name, 0)
        except IndexError: continue
        try: l2, b2 = get_pairs(i, name, 1)
        except IndexError: continue

        mt1, mw1 = get_masses(nu1, b1, l1)
        mt2, mw2 = get_masses(nu2, b2, l2)

        # ----- cuda neutrinos ------ #
        snu1, snu2 = get_static(i, name)
        dnu1, dnu2 = get_dynamic(i, name)

        # ----- Static Masses ------ #
        snu = DoubleNu((b1, b2), (l1, l2), i, mW, mT, mW, mT).nunu_s
        snu1_, snu2_ = get_best({0 : nu1, 1 : nu2}, snu)

        # ----- Dynamic Masses ----- #
        dnu = DoubleNu((b1, b2), (l1, l2), i, mw1, mt1, mw2, mt2).nunu_s
        dnu1_, dnu2_ = get_best({0 : nu1, 1 : nu2}, dnu)

        con = Container(
                make_types((nu1  , nu2  ), (l1, l2), (b1, b2)), # truths
                make_types((dnu1 , dnu2 ), (l1, l2), (b1, b2)), # cuda_dyn
                make_types((snu1 , snu2 ), (l1, l2), (b1, b2)), # cuda_static
                make_types((dnu1_, dnu2_), (l1, l2), (b1, b2)), # reference dynamic
                make_types((snu1_, snu2_), (l1, l2), (b1, b2))  # reference static
        )
        con.event_data["met"] = i.met
        con.event_data["phi"] = i.phi
        conx[ix] = con 
        update = True
        if ix % 1000 != 999: continue
        pickle.dump(conx, open(name + ".pkl", "wb"))
    if not update: return conx
    pickle.dump(conx, open(name + ".pkl", "wb"))
    return conx

def compxl(sl):
    return {
            "topchildren" : compiler(sl.Events, "top_children"),
            "truthjet"    : compiler(sl.Events, "truthjet"),
            "jetchildren" : compiler(sl.Events, "jetchildren"),
            "jetleptons"  : compiler(sl.Events, "jetleptons")
    }

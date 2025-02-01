from .nusol import DoubleNu, Particle, event
import numpy
import pickle

def chi2(tru, reco):
    dx = (tru.px - reco.px)/1000
    dy = (tru.py - reco.py)/1000
    dz = (tru.pz - reco.pz)/1000
    return dx**2 + dy**2 + dz**2

def MakeParticle(p):
    x = []
    for i in p: x.append(Particle(i.pt, i.eta, i.phi, i.e))
    return x

def record_data(inpt, truth, data, leps, bqrk):
    if not len(inpt): return 1
    for i in range(len(inpt)):
        kx = "n" + str(i+1)
        if truth is not None: tr = truth[i]
        else: tr = None

        nu = inpt[i]
        lp = leps[i]
        bq = bqrk[i]

        data["px"][kx]   += [nu.px / 1000]
        data["py"][kx]   += [nu.py / 1000]
        data["pz"][kx]   += [nu.pz / 1000]
        data["tmass"][kx] += [(lp + bq + nu).Mass / 1000]
        data["wmass"][kx] += [(lp + nu).Mass / 1000]

        if tr is None: continue
        data["chi2"][kx] += [chi2(tr, nu)]
        if i: continue
        try: data["dst"] += [nu.distance]
        except AttributeError: pass
    return 0

def makeData():
    return {
        "missed": 0, "dst" : [],
        "chi2" : {"n1" : [], "n2" : []},
        "px"   : {"n1" : [], "n2" : []},
        "py"   : {"n1" : [], "n2" : []},
        "pz"   : {"n1" : [], "n2" : []},
        "tmass": {"n1" : [], "n2" : []},
        "wmass": {"n1" : [], "n2" : []}
    }

def makeTruth():
    return {
        "px"   : {"n1" : [], "n2" : []},
        "py"   : {"n1" : [], "n2" : []},
        "pz"   : {"n1" : [], "n2" : []},
        "tmass": {"n1" : [], "n2" : []},
        "wmass": {"n1" : [], "n2" : []}
    }

def reload(fname): return pickle.load(open("./data/" + fname +".pkl", "rb"))

def compile_neutrinos(ana = None, truth_top = None, truth_lep = None, truth_b = None, truth_w = None, reco_c1 = None, reco_c2 = None, fname = None):
    if ana is None: return reload(fname)

    truth_nu = ana.truth_nus
    mt = 172.62 * 1000
    mw = 80.385 * 1000

    met = ana.met
    phi = ana.phi

    r1_cu = makeData() # injected truth masses
    r2_cu = makeData() # static masses
    r1_rf = makeData() # inject truth masses
    r2_rf = makeData() # static masses
    truth_nux = makeTruth()

    dt = {"i" : 0, "r1_cu" : r1_cu, "r2_cu" : r2_cu, "r1_rf" : r1_rf, "r2_rf" : r2_rf, "truth_nux" : truth_nux}
    try:
        dt = reload(fname)
        r1_cu, r2_cu = dt["r1_cu"], dt["r2_cu"]
        r1_rf, r2_rf = dt["r1_rf"], dt["r2_rf"]
        truth_nux = dt["truth_nux"]
    except: pass

    update = False
    for i in range(len(truth_nu)):
        if dt["i"] > i: continue
        tru_nunu = truth_nu[i]
        tru_top  = truth_top[i]
        tru_lep  = truth_lep[i]
        tru_b    = truth_b[i]
        tru_w    = truth_w[i]
        if not len(tru_b) or not len(tru_lep): continue

        r1_nunu = reco_c1[i]
        r2_nunu = reco_c2[i]

        b1, b2 = MakeParticle(tru_b)
        l1, l2 = MakeParticle(tru_lep)
        ev = event(met[i], phi[i])

        r1_cu["missed"] += record_data(r1_nunu, tru_nunu, r1_cu, tru_lep, tru_b)
        r2_cu["missed"] += record_data(r2_nunu, tru_nunu, r2_cu, tru_lep, tru_b)

        try: nunu = DoubleNu((b1, b2), (l1, l2), ev, tru_w[0].Mass, tru_top[0].Mass, tru_w[1].Mass, tru_top[1].Mass)
        except numpy.linalg.LinAlgError: nunu = None
        except ValueError: nunu = None

        if nunu is not None: nunu = nunu.nunu_s
        else: nunu = []
        r1_rf["missed"] += record_data(nunu, tru_nunu, r1_rf, tru_lep, tru_b)

        try: nunu = DoubleNu((b1, b2), (l1, l2), ev, mw, mt, mw, mt)
        except numpy.linalg.LinAlgError: nunu = None
        except ValueError: nunu = None

        if nunu is not None: nunu = nunu.nunu_s
        else: nunu = []
        r2_rf["missed"] += record_data(nunu, tru_nunu, r2_rf, tru_lep, tru_b)

        record_data(tru_nunu, None, truth_nux, tru_lep, tru_b)

        print(i, len(truth_nu))
        if i % 50 != 49: continue
        dt["i"] = i
        f = open("./data/" + fname +".pkl", "wb")
        pickle.dump(dt, f)
        f.close()
        update = True

    if update:
        dt["i"] = len(truth_nu)
        f = open("./data/" + fname +".pkl", "wb")
        pickle.dump(dt, f)
        f.close()
    return dt


def topchildren_nunu_build(ana = None):
    if ana is not None:
        truth_top  = ana.truth_tops
        truth_lep  = ana.truth_leptons
        truth_b    = ana.truth_bquarks
        truth_w    = ana.truth_bosons
        # -------------- #

        reco_c1  = ana.c1_reconstructed_children_nu
        reco_c2  = ana.c2_reconstructed_children_nu
        return compile_neutrinos(ana, truth_top, truth_lep, truth_b, truth_w, reco_c1, reco_c2, "neutrino-children")
    return compile_neutrinos(ana, fname = "neutrino-children")


def toptruthjets_nunu_build(ana = None):
    if ana is not None:
        truth_top  = ana.truth_jets_top
        truth_lep  = ana.truth_leptons
        truth_b    = ana.truth_bjets
        truth_w    = ana.truth_bosons
        # -------------- #

        reco_c1  = ana.c1_reconstructed_truthjet_nu
        reco_c2  = ana.c2_reconstructed_truthjet_nu
        return compile_neutrinos(ana, truth_top, truth_lep, truth_b, truth_w, reco_c1, reco_c2, "neutrino-truthjets")
    return compile_neutrinos(ana, fname = "neutrino-truthjets")

def topjetchild_nunu_build(ana = None):
    if ana is not None:
        truth_top  = ana.jets_top
        truth_lep  = ana.truth_leptons
        truth_b    = ana.bjets
        truth_w    = ana.truth_bosons
        # -------------- #

        reco_c1  = ana.c1_reconstructed_jetchild_nu
        reco_c2  = ana.c2_reconstructed_jetchild_nu
        return compile_neutrinos(ana, truth_top, truth_lep, truth_b, truth_w, reco_c1, reco_c2, "neutrino-jetchild")
    return compile_neutrinos(ana, fname = "neutrino-jetchild")


def topdetector_nunu_build(ana = None):
    if ana is not None:
        truth_top  = ana.lepton_jets_top
        truth_lep  = ana.reco_leptons
        truth_b    = ana.bjets
        truth_w    = ana.reco_bosons
        # -------------- #

        reco_c1  = ana.c1_reconstructed_jetlep_nu
        reco_c2  = ana.c2_reconstructed_jetlep_nu
        return compile_neutrinos(ana, truth_top, truth_lep, truth_b, truth_w, reco_c1, reco_c2, "neutrino-detector")
    return compile_neutrinos(ana, fname = "neutrino-detector")


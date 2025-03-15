from AnalysisG.core.plotting import TH1F, TH2F

def path(hist, subx = ""):
    hist.Style = "ATLAS"
    hist.Filename = subx
    hist.OutputDirectory = "Figures/neutrino-combinatorial/"
    return hist

def mapping(l):
    if l == 11: return "$e$"
    if l == 13: return "$\\mu$"
    if l == 15: return "$\\tau$"
    if l == -11: return "$\\bar{e}$"
    if l == -13: return "$\\bar{\\mu}$"
    if l == -15: return "$\\bar{\\tau}$"
    return "NA"

def GetMissingEnergy(data):
    out = {"children" : [], "neutrino" : [], "observed_met" : [], "neutrino_met" : []}
    for i, ev in data.events.items():
        out["children"].append(ev.delta_met)
        out["neutrino"].append(ev.delta_metnu)
        out["observed_met"].append(ev.observed_met)
        out["neutrino_met"].append(ev.neutrino_met)
    return out

def GetMasses(data):
    def chi2(p1, p2): return (p1.px - p2.px)**2 + (p1.py - p2.py)**2 + (p1.pz - p2.pz)**2
    def chi2_(reco, truth):
        chi2x = {}
        for i in range(len(reco)): chi2x[(chi2(reco[i][0], truth[0]) + chi2(reco[i][1], truth[1]))**0.5] = reco[i]
        if not len(chi2x): return []
        return chi2x[sorted(chi2x)[0]]

    out = {"top-mass-child" : [], "top-mass-cobs" : [], "top-mass-cmet" : [], "top-mass-robs" : [], "top-mass-rmet" : []}
    for i, ev in data.events.items():
        cnus = ev.cobs_neutrinos
        cmet = ev.cmet_neutrinos

        rnus = ev.robs_neutrinos
        rmet = ev.rmet_neutrinos
        trun = ev.truth_neutrinos

        bqrk = ev.bquark
        lept = ev.lepton


        out["top-mass-child"] += [t.Mass / 1000 for t in ev.tops]

        cnus = chi2_(cnus, trun)
        out["top-mass-cobs"]  += [(cnus[x].bquark + cnus[x].lepton + cnus[x]).Mass / 1000 for x in range(len(cnus))]

        cmet = chi2_(cmet, trun)
        out["top-mass-cmet"]  += [(cmet[x].bquark + cmet[x].lepton + cmet[x]).Mass / 1000 for x in range(len(cmet))]

        rnus = chi2_(rnus, trun)
        out["top-mass-robs"] += [(rnus[t] + bqrk[t] + lept[t]).Mass / 1000 for t in range(len(rnus))]

        rmet = chi2_(rmet, trun)
        out["top-mass-rmet"] += [(rmet[t] + bqrk[t] + lept[t]).Mass / 1000 for t in range(len(rmet))]
    return out

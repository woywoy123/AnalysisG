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
    out = {"top-mass-child" : [], "top-mass-cobs" : [], "top-mass-cmet" : [], "top-mass-robs" : [], "top-mass-rmet" : []}
    for i, ev in data.events.items():
        cnus = ev.cobs_neutrinos
        cmet = ev.cmet_neutrinos

        rnus = ev.robs_neutrinos
        rmet = ev.rmet_neutrinos

        bqrk = ev.bquark
        lept = ev.lepton

        out["top-mass-child"] += [t.Mass / 1000 for t in ev.tops]
        out["top-mass-cobs"]  += [(cnus[t] + bqrk[t] + lept[t]).Mass / 1000 for t in range(len(cnus))]
        out["top-mass-cmet"]  += [(cmet[t] + bqrk[t] + lept[t]).Mass / 1000 for t in range(len(cmet))]

        out["top-mass-robs"]  += [(rnus[t] + bqrk[t] + lept[t]).Mass / 1000 for t in range(len(rnus))]
        out["top-mass-rmet"]  += [(rmet[t] + bqrk[t] + lept[t]).Mass / 1000 for t in range(len(rmet))]

    return out

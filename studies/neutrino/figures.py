from AnalysisG.core.plotting import TH1F, TH2F, TLine, ROC
from AnalysisG.core import *
import math

def path(hist, subx = "", defaults = None):
    hist.UseLateX = False
    hist.Style = "ATLAS"
    hist.OutputDirectory = "Figures" + subx
    return hist

def double_neutrino(data, total = False):
    def build_hist(datax, dicx):
        hists = []
        for name, keys in dicx.items():
            f = TH1F()
            f.Density = True
            f.Title = name
            f.Color = ["red", "blue", "green"][len(hists)%3]
            f.xData = datax[keys]
            hists.append(f)
        return hists

    def mapping(l):
        if l == 11: return "$e$"
        if l == 13: return "$\\mu$"
        if l == 15: return "$\\tau$"
        if l == -11: return "$\\bar{e}$"
        if l == -13: return "$\\bar{\\mu}$"
        if l == -15: return "$\\bar{\\tau}$"
        return "NA"

    def tag(l):
        out = ""
        for k in l.split(" "):
            if k == "$e$":            out += "e"
            if k == "$\\mu$":         out += "mu"
            if k == "$\\tau$":        out += "tau"
            if k == "$\\bar{e}$":     out += "be"
            if k == "$\\bar{\\mu}$":  out += "bmu"
            if k == "$\\bar{\\tau}$": out += "btau"
            out += "_"
        return out

    dist_nu     = data.dist_nu
    pdgids      = data.pdgid

    tru_topmass = data.tru_topmass
    exp_topmass = data.exp_topmass
    nusol_tmass = data.nusol_tmass

    tru_wmass   = data.tru_wmass
    exp_wmass   = data.exp_wmass
    nusol_wmass = data.nusol_wmass

    wmass_pdgid   = {}
    topmass_pdgid = {}
    nusol_pdgid   = {}

    for i in list(dist_nu):
        if len([k for k in tru_topmass[i] if abs(k-172.62) < 2]) != 2: continue
        lepsym = " ".join([mapping(l) for l in sorted(pdgids[i])])
        if lepsym not in topmass_pdgid: topmass_pdgid[lepsym] = {"truth" : [], "expected" : [], "nusol" : []}
        if lepsym not in   wmass_pdgid: wmass_pdgid[lepsym]   = {"truth" : [], "expected" : [], "nusol" : []}
        if lepsym not in   nusol_pdgid: nusol_pdgid[lepsym]   = []
        if i != "0xd0edab7810edc54f": continue
        print(i, tru_topmass[i], exp_topmass[i], tru_wmass[i], exp_wmass[i])
        exit()

        wmass_pdgid[lepsym]["truth"]   += tru_wmass[i]
        topmass_pdgid[lepsym]["truth"] += tru_topmass[i]

        wmass_pdgid[lepsym]["expected"]   += exp_wmass[i]
        topmass_pdgid[lepsym]["expected"] += exp_topmass[i]

        wmass_pdgid[lepsym]["nusol"]   += nusol_wmass[i]
        topmass_pdgid[lepsym]["nusol"] += nusol_tmass[i]

        nusol_pdgid[lepsym] += [-math.log(dist_nu[i], 10)]

    keys = {"Truth" : "truth", "Bruteforced" : "expected", "Solution" : "nusol"}

    sol_hist = []
    for i in list(nusol_pdgid):
        sol_dist = TH1F()
        sol_dist.Title = i
        sol_dist.Density = True
        sol_dist.xData = nusol_pdgid[i]
        sol_hist.append(sol_dist)
        if total: continue

        hists = build_hist(topmass_pdgid[i], keys)
        th = path(TH1F(), "/top_nunu")
        th.Title  = "Comparison of Invariant Top Mass and Double Neutrino Algorithm (" + i + ")"
        th.xTitle = "Invariant Mass (GeV)"
        th.yTitle = "Leptonic Top / 0.1 GeV"
        th.Histograms = hists
        th.xBins  = 400
        th.xStep  = 5
        th.xMin   = 150
        th.xMax   = 190
        th.ShowCount = True
        th.Filename = "top_nunu-" + tag(i)
        th.SaveFigure()

        hists = build_hist(wmass_pdgid[i], keys)
        th = path(TH1F(), "/w-boson")
        th.Title  = "Comparison of Invariant W-Boson Mass and Double Neutrino Algorithm (" + i + ")"
        th.xTitle = "Invariant Mass (GeV)"
        th.yTitle = "Leptonic W-Boson / 1 GeV"
        th.Histograms = hists
        th.xBins  = 800
        th.xStep  = 5
        th.xMin   = 40
        th.xMax   = 120
        th.ShowCount = True
        th.Filename = "w_nunu-" + tag(i)
        th.SaveFigure()

    th = path(TH1F())
    th.Title  = "Double Neutrino Ellipse Solution Distance"
    th.xTitle = "$-\\log(solution)$ (Arb.)"
    th.yTitle = "Events / (1)"
    th.Histograms = sol_hist
    th.xBins  = 40
    th.xStep  = 4
    th.xMin   = 0
    th.xMax   = 40
    th.ShowCount = True
    th.Filename = "nunu-sol"
    th.SaveFigure()

    hists = []
    for name, keyx in keys.items():
        f = TH1F()
        f.Density = True
        f.Color = ["red", "blue", "green"][len(hists)%3]
        f.Title = name
        f.xData = sum([topmass_pdgid[i][keyx] for i in list(nusol_pdgid)], [])
        hists.append(f)

    th = path(TH1F())
    th.Title  = "Comparison of Invariant Top Mass and Double Neutrino Algorithm"
    th.xTitle = "Invariant Mass (GeV)"
    th.yTitle = "Leptonic Top / 1 GeV"
    th.Histograms = hists
    th.xBins  = 400
    th.xStep  = 5
    th.xMin   = 150
    th.xMax   = 190
    th.ShowCount = True
    th.Filename = "top_nunu_total"
    th.SaveFigure()

    hists = []
    for name, keyx in keys.items():
        f = TH1F()
        f.Density = True
        f.Color = ["red", "blue", "green"][len(hists)%3]
        f.Title = name
        f.xData = sum([wmass_pdgid[i][keyx] for i in list(nusol_pdgid)], [])
        hists.append(f)

    th = path(TH1F())
    th.Title  = "Comparison of Invariant W-Boson Mass and Double Neutrino Algorithm"
    th.xTitle = "Invariant Mass (GeV)"
    th.yTitle = "Leptonic W-Boson / 1 GeV"
    th.Histograms = hists
    th.xBins  = 800
    th.xStep  = 5
    th.xMin   = 40
    th.xMax   = 120
    th.ShowCount = True
    th.Filename = "w_nunu_total"
    th.SaveFigure()



def missing_energy(data):

    delta_met   = data.delta_met   # obs - summing all children
    delta_metnu = data.delta_metnu # obs - summing only neutrinos

    nus_met     = data.nus_met     # raw missing et from neutrinos
    obs_met     = data.obs_met     # raw missing et observed.

    met_obs = TH1F()
    met_obs.Title = "Observed"
    met_obs.xData = list(obs_met.values())

    met_nus = TH1F()
    met_nus.Title = "$\\sum \\nu$"
    met_nus.xData = list(nus_met.values())

    th = path(TH1F())
    th.Title  = "Comparison of Missing Transverse Energy from Observation and Neutrinos"
    th.xTitle = "Missing Transverse Energy (GeV)"
    th.yTitle = "Events / 5 GeV"
    th.Histogram  = met_obs
    th.Histograms = [met_nus]
    th.xBins  = 200
    th.xStep  = 100
    th.xMin   = 0
    th.xMax   = 1000
    th.Stacked   = True
    th.ShowCount = True
    th.Filename = "met_comparison"
    th.SaveFigure()


    met_obs = TH1F()
    met_obs.Title = "$\\text{obs} - \\sum \\text{partons}_{\\text{top}}$"
    met_obs.xData = list(delta_met.values())

    met_nus = TH1F()
    met_nus.Title = "$\\text{obs} - \\sum \\nu_{\\text{top}}$"
    met_nus.xData = list(delta_metnu.values())

    th = path(TH1F())
    th.Title  = "Observed Missing Transverse Energy Differential between Event Partons and Neutrinos"
    th.xTitle = "$\\Delta \\text{MET}$ (GeV)"
    th.yTitle = "Events / 5 GeV"
    th.Histogram  = met_obs
    th.Histograms = [met_nus]
    th.xBins  = 200
    th.xStep  = 100
    th.xMin   = 0
    th.xMax   = 1000
    th.Stacked   = True
    th.ShowCount = True
    th.Filename = "met_delta"
    th.SaveFigure()





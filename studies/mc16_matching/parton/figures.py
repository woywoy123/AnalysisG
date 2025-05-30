from AnalysisG.core.plotting import TH1F, TH2F

global figure_path
global mass_point

colors = ["blue", "red", "green", "orange", "pink", "yellow"]
pdgid = {"1" : "light-quarks", "2" : "light-quarks", "3" : "light-quarks", "4" : "light-quarks", "5" : "b-quark", "21" : "gluon"}

def path(hist):
    hist.Style = "ATLAS"
    hist.OutputDirectory = figure_path + "/parton/" + mass_point
    return hist

def ntops_tjets_pt(ana, col):

    pt = ana.ntops_tjets_pt
    hists = []
    for i in pt:
        if i.split("::")[0] == "0": continue
        th = TH1F()
        th.xData = pt[i]
        th.Title = i.split("::")[0] + "-Tops"
        th.Density = False
        th.Color = next(col)
        th.Alpha = 0.8
        hists.append(th)

    th_tj = path(TH1F())
    th_tj.Title = "Truth Jet $p_T$ Containing n-Top Contributions"
    th_tj.Histograms = hists
    th_tj.xMin = 0
    th_tj.xMax = 1200
    th_tj.xBins = 300
    th_tj.xStep = 100
    th_tj.Stacked = True
    th_tj.yLogarithmic = True
    th_tj.yTitle = "Truth Jets / (4 GeV)"
    th_tj.xTitle = "Truth Jet $p_T$ (GeV)"
    th_tj.Filename = "Figure.8.a"
    th_tj.SaveFigure()


def ntops_tjets_e(ana, col):

    pt = ana.ntops_tjets_e
    hists = []
    for i in pt:
        if i.split("::")[0] == "0": continue
        th = TH1F()
        th.xData = pt[i]
        th.Title = i.split("::")[0] + "-Tops"
        th.Color = next(col)
        th.Alpha = 0.8
        hists.append(th)

    th_tj = path(TH1F())
    th_tj.Title = "Truth Jet Energy Containing n-Top Contributions"
    th_tj.Histograms = hists
    th_tj.xMin = 0
    th_tj.xMax = 1200
    th_tj.xBins = 300
    th_tj.xStep = 100
    th_tj.Stacked = True
    th_tj.yLogarithmic = True
    th_tj.yTitle = "Truth Jets / (4 GeV)"
    th_tj.xTitle = "Truth Jet Energy (GeV)"
    th_tj.Filename = "Figure.8.b"
    th_tj.SaveFigure()

def ntops_jets_pt(ana, col):

    pt = ana.ntops_jets_pt
    hists = []
    for i in pt:
        if i.split("::")[0] == "0": continue
        th = TH1F()
        th.xData = pt[i]
        th.Title = i.split("::")[0] + "-Tops"
        th.Color = next(col)
        th.Alpha = 0.8
        hists.append(th)

    th_tj = path(TH1F())
    th_tj.Title = "Jet $p_T$ Containing n-Top Contributions"
    th_tj.Histograms = hists
    th_tj.xMin = 0
    th_tj.xMax = 1200
    th_tj.xBins = 300
    th_tj.xStep = 100
    th_tj.Stacked = True
    th_tj.yTitle = "Jets / (4 GeV)"
    th_tj.xTitle = "Jet $p_T$ (GeV)"
    th_tj.Filename = "Figure.8.c"
    th_tj.SaveFigure()

def ntops_jets_e(ana, col):

    pt = ana.ntops_jets_e
    hists = []
    for i in pt:
        if i.split("::")[0] == "0": continue
        th = TH1F()
        th.xData = pt[i]
        th.Title = i.split("::")[0] + "-Tops"
        th.Color = next(col)
        th.Alpha = 0.8
        hists.append(th)

    th_tj = path(TH1F())
    th_tj.Title = "Jet Energy Containing n-Top Contributions"
    th_tj.Histograms = hists
    th_tj.xMin = 0
    th_tj.xMax = 1200
    th_tj.xBins = 300
    th_tj.xStep = 100
    th_tj.Stacked = True
    th_tj.yLogarithmic = True
    th_tj.yTitle = "Jets / (4 GeV)"
    th_tj.xTitle = "Jet Energy (GeV)"
    th_tj.Filename = "Figure.8.d"
    th_tj.SaveFigure()


def nparton_tjets_e(ana, col):
    pt = ana.nparton_tjet_e
    hists = {}
    for i in pt:
        n = int(i.split("::")[0])
        if not n: continue
        if not len(pt[i]): continue

        if n >= 4: tl = "$\\geq 4$-Partons"
        else: tl = str(n) + "-Partons"
        if tl not in hists: hists[tl] = pt[i]
        else: hists[tl] += pt[i]

    for i in hists:
        th = TH1F()
        th.xData = hists[i]
        th.Title = i
        th.Color = next(col)
        th.Alpha = 0.8
        hists[i] = th

    th_tj = path(TH1F())
    th_tj.Title = "Parton Content to Truth Jet Energy Ratio for n-Partons Matched"
    th_tj.Histograms = list(hists.values())[1:] + [list(hists.values())[0]]
    th_tj.xMin = 0
    th_tj.xMax = 2.01
    th_tj.xBins = 201
    th_tj.xStep = 0.1
    th_tj.Stacked = True
    th_tj.yTitle = "Jets / 0.01"
    th_tj.xTitle = "Parton Content to Truth Jet Energy Ratio ($\\sum^{n}_{i} \\text{p}^i_{e} / \\text{truth jet}_{e}$)"
    th_tj.Filename = "Figure.8.e"
    th_tj.SaveFigure()

def nparton_jets_e(ana, col):
    pt = ana.nparton_jet_e
    hists = {}
    for i in pt:
        n = int(i.split("::")[0])
        if not n: continue
        if not len(pt[i]): continue

        if n >= 4: tl = "$\\geq 4$-Partons"
        else: tl = str(n) + "-Partons"
        if tl not in hists: hists[tl] = pt[i]
        else: hists[tl] += pt[i]

    for i in hists:
        th = TH1F()
        th.xData = hists[i]
        th.Title = i
        th.Color = next(col)
        th.Alpha = 0.8
        hists[i] = th

    th_tj = path(TH1F())
    th_tj.Title = "Parton Content to Jet Energy Ratio for n-Partons Matched"
    th_tj.Histograms = list(hists.values())[1:] + [list(hists.values())[0]]
    th_tj.xMin = 0
    th_tj.xMax = 2.01
    th_tj.xBins = 201
    th_tj.xStep = 0.1
    th_tj.Stacked = True
    th_tj.yTitle = "Jets / 0.01"
    th_tj.xTitle = "Parton Content to Jet Energy Ratio ($\\sum^{n}_{i} \\text{p}^i_{e} / \\text{jet}_{e}$)"
    th_tj.Filename = "Figure.8.f"
    th_tj.SaveFigure()


def nparton_tjets_pdgid_e(ana, col):
    pt = ana.frac_parton_tjet_e
    hists = {}
    for i in pt:
        ix = pdgid[i]
        if ix not in hists: hists[ix] = pt[i]
        else: hists[ix] += pt[i]

    for i in hists:
        th = TH1F()
        th.xData = hists[i]
        th.Title = i
        th.Color = next(col)
        th.Alpha = 0.8
        hists[i] = th

    th_tj = path(TH1F())
    th_tj.Title = "Fractional Parton Energy Contribution to Matched Truth Jet"
    th_tj.Histograms = list(hists.values())
    th_tj.xMin = 0
    th_tj.xMax = 1.01
    th_tj.xBins = 101
    th_tj.xStep = 0.1
    th_tj.Stacked = True
    th_tj.yTitle = "Partons / 0.01"
    th_tj.xTitle = "Fractional Parton Contribution ($p_{e} / \\sum^{n}_{i} \\text{p}^{i}_{e}$)"
    th_tj.Filename = "Figure.8.g"
    th_tj.SaveFigure()

def nparton_jets_pdgid_e(ana, col):
    pt = ana.frac_parton_jet_e
    hists = {}
    for i in pt:
        ix = pdgid[i]
        if ix not in hists: hists[ix] = pt[i]
        else: hists[ix] += pt[i]

    for i in hists:
        th = TH1F()
        th.xData = hists[i]
        th.Title = i
        th.Color = next(col)
        th.Alpha = 0.8
        hists[i] = th

    th_tj = path(TH1F())
    th_tj.Title = "Fractional Parton Energy Contribution to Matched Jet"
    th_tj.Histograms = list(hists.values())
    th_tj.xMin = 0
    th_tj.xMax = 1.01
    th_tj.xBins = 101
    th_tj.xStep = 0.1
    th_tj.Stacked = True
    th_tj.yTitle = "Partons / 0.01"
    th_tj.xTitle = "Fractional Parton Contribution ($p_{e} / \\sum^{n}_{i} \\text{p}_{e}^{i}$)"
    th_tj.Filename = "Figure.8.h"
    th_tj.SaveFigure()


def frac_ntop_tjet_contrib(ana, col):
    pt = ana.frac_ntop_tjet_contribution
    hists = {}
    for i in pt:
        th = TH1F()
        th.xData = pt[i]
        th.Title = i.split("::")[0] + "-Tops"
        th.Color = next(col)
        th.Alpha = 0.8
        hists[i] = th

    th_tj = path(TH1F())
    th_tj.Title = "Parton Energy Contribution of Top-Quark for n-Top Matched Truth Jets"
    th_tj.Histograms = list(hists.values())
    th_tj.xMin = 0
    th_tj.xMax = 2.01
    th_tj.xBins = 201
    th_tj.xStep = 0.1
    th_tj.Stacked = True
    th_tj.yTitle = "Partons / 0.01"
    th_tj.xTitle = "Fractional Parton Contribution ($p_{e} / \\text{truth jet}^{\\text{n-top}}_{e}$)"
    th_tj.Filename = "Figure.8.i"
    th_tj.SaveFigure()

def frac_ntop_jet_contrib(ana, col):
    pt = ana.frac_ntop_jet_contribution
    hists = {}
    for i in pt:
        th = TH1F()
        th.xData = pt[i]
        th.Title = i.split("::")[0] + "-Tops"
        th.Color = next(col)
        th.Alpha = 0.8
        hists[i] = th

    th_tj = path(TH1F())
    th_tj.Title = "Parton Energy Contribution of Top-Quark for n-Top Matched Jets"
    th_tj.Histograms = list(hists.values())
    th_tj.xMin = 0
    th_tj.xMax = 2.01
    th_tj.xBins = 201
    th_tj.xStep = 0.1
    th_tj.Stacked = True
    th_tj.yTitle = "Partons / 0.01"
    th_tj.xTitle = "Fractional Parton Contribution ($p_{e} / \\text{jet}^{\\text{n-top}}_{e}$)"
    th_tj.Filename = "Figure.8.j"
    th_tj.SaveFigure()

def top_mass_wth_cuts(ana, col):
    pt = ana.frac_mass_top
    hists = []
    for i in pt:
        thx = i.split("::")[0]
        if "::jet" not in i: continue
        if "::jet-gluon" in i: continue
        if "0.00" == thx: tlt = "No threshold"
        elif "0.10" == thx: tlt = "$\\geq$" + thx
        elif "0.20" == thx: tlt = "$\\geq$" + thx
        elif "0.30" == thx: tlt = "$\\geq$" + thx
        elif "0.40" == thx: tlt = "$\\geq$" + thx
        elif "0.50" == thx: tlt = "$\\geq$" + thx
        elif "0.60" == thx: tlt = "$\\geq$" + thx
        elif "0.70" == thx: tlt = "$\\geq$" + thx
        elif "0.80" == thx: tlt = "$\\geq$" + thx
        elif "0.90" == thx: tlt = "$\\geq$" + thx
        elif "1.00" == thx: tlt = "$\\geq$" + thx
        elif "1.10" == thx: tlt = "$\\geq$" + thx
        else: continue

        th = TH1F()
        th.xData = pt[i]
        th.Title = tlt
        th.Alpha = 0.5
        #th.Density = True
        hists.append(th)

    for i in range(1, len(hists)):
        hists[0].Color = "red"
        hists[i].Color = "blue"
        th_tj = path(TH1F())
        th_tj.Title = "Invariant Top Mass at Constituent Threshold " + hists[i].Title + " for Jets"
        th_tj.Histograms = [hists[0], hists[i]]
        th_tj.xMin = 0
        th_tj.xMax = 400
        th_tj.xBins = 200
        th_tj.xStep = 40
        th_tj.Alpha = 0.5
        th_tj.Overflow = False
        th_tj.yTitle = "Tops / 1 (GeV)"
        th_tj.xTitle = "Invariant Top Mass (GeV)"
        th_tj.Filename = "Figure.8.k-" + str(i)
        th_tj.SaveFigure()


def top_mass_wth_cuts_gluons(ana, col):
    pt = ana.frac_mass_top
    hists = []
    hist_gl = [None, None]
    for i in pt:
        thx = i.split("::")[0]
        if "::jet-gluon" not in i: continue

        if "0.00" == thx and "rm"  in i: tlt = "No threshold (All Gluon Jets Removed)"
        elif "0.00" == thx and "inc" in i: tlt = "No threshold (All Gluon Jets)"
        elif "0.10" == thx: tlt = "$\\geq$" + thx + " (Threshold on Gluon Jets)"
        elif "0.20" == thx: tlt = "$\\geq$" + thx + " (Threshold on Gluon Jets)"
        elif "0.30" == thx: tlt = "$\\geq$" + thx + " (Threshold on Gluon Jets)"
        elif "0.40" == thx: tlt = "$\\geq$" + thx + " (Threshold on Gluon Jets)"
        elif "0.50" == thx: tlt = "$\\geq$" + thx + " (Threshold on Gluon Jets)"
        elif "0.60" == thx: tlt = "$\\geq$" + thx + " (Threshold on Gluon Jets)"
        elif "0.70" == thx: tlt = "$\\geq$" + thx + " (Threshold on Gluon Jets)"
        elif "0.80" == thx: tlt = "$\\geq$" + thx + " (Threshold on Gluon Jets)"
        elif "0.90" == thx: tlt = "$\\geq$" + thx + " (Threshold on Gluon Jets)"
        elif "1.00" == thx: tlt = "$\\geq$" + thx + " (Threshold on Gluon Jets)"
        elif "1.10" == thx: tlt = "$\\geq$" + thx + " (Threshold on Gluon Jets)"
        else: continue


        th = TH1F()
        th.xData = pt[i]
        th.Title = tlt
        th.Alpha = 0.5

        if   "0.00" == thx and "rm" in i:  hist_gl[0] = th
        elif "0.00" == thx and "inc" in i: hist_gl[1] = th
        else: hists.append(th)

    for i in range(0, len(hists)):
        if i == 0: th = hist_gl[0]
        else: th = hist_gl[1]

        th.Color = "red"
        hists[i].Color = "blue"
        th_tj = path(TH1F())
        th_tj.Title = "Invariant Top Mass for Gluon Jet Constituent Threshold " + hists[i].Title
        th_tj.Histograms = [th, hists[i]]
        th_tj.xMin = 0
        th_tj.xMax = 400
        th_tj.xBins = 200
        th_tj.xStep = 40
        th_tj.Alpha = 0.5
        th_tj.Overflow = False
        th_tj.yTitle = "Tops / 1 (GeV)"
        th_tj.xTitle = "Invariant Top Mass (GeV)"
        th_tj.Filename = "Figure.8.p-" + str(i)
        th_tj.SaveFigure()


def Parton(ana):
    #ntops_tjets_pt(ana , iter(colors))
    #ntops_tjets_e(ana  , iter(colors))
    #ntops_jets_pt(ana  , iter(colors))
    #ntops_jets_e(ana   , iter(colors))
    #nparton_tjets_e(ana, iter(colors))
    #nparton_jets_e(ana , iter(colors))
    #nparton_tjets_pdgid_e(ana, iter(colors))
    #nparton_jets_pdgid_e(ana, iter(colors))
    #frac_ntop_tjet_contrib(ana, iter(colors))
    #frac_ntop_jet_contrib(ana, iter(colors))
    top_mass_wth_cuts(ana, iter(colors))
    top_mass_wth_cuts_gluons(ana, iter(colors))



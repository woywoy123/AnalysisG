from AnalysisG.core.plotting import TH1F, TH2F

global figure_path
global mass_point

def path(hist):
    hist.OutputDirectory = figure_path + "/top-matching/" + mass_point
    hist.Style = "ATLAS"
    return hist

def top_matching(ana):

    tru_top = TH1F()
    tru_top.Title = "Truth Top"
    tru_top.xData = ana.truth_top
    tru_top.Color = "red"

    tru_ch = TH1F()
    tru_ch.Title = "Top Children"
    tru_ch.xData = ana.truth_children["all"]
    tru_ch.Color = "black"

    tru_tj = TH1F()
    tru_tj.Title = "Truth-Jets (Truth $\\ell + \\nu$)"
    tru_tj.xData = ana.truth_jets["all"]
    tru_tj.Color = "blue"

    tru_j = TH1F()
    tru_j.Title = "Jets (Truth $\\ell + \\nu$)"
    tru_j.xData = ana.jets_truth_leps["all"]
    tru_j.Color = "orange"

    tru_jl = TH1F()
    tru_jl.Title = "Jets Leptons (Truth $\\nu$)"
    tru_jl.xData = ana.jet_leps["all"]
    tru_jl.Color = "pink"

    all_t = path(TH1F())
    all_t.Histograms = [tru_top, tru_j, tru_tj, tru_jl, tru_ch]
    all_t.Title = "Matching Scheme for Tops using Varying Level of Truth Information"
    all_t.xTitle = "Invariant Mass of Matched Top (GeV)"
    all_t.yTitle = "Density (Arb.) / 1 GeV"
    all_t.xMin = 0
    all_t.xMax = 400
    all_t.xBins = 400
    all_t.xStep = 20
    all_t.Density = True
    all_t.Overflow = False
    all_t.yLogarithmic = True
    all_t.Filename = "Figure.2.a"
    all_t.SaveFigure()

def top_decay_channel_children(ana):

    lep_ch = TH1F()
    lep_ch.Title = "Leptonic"
    lep_ch.xData = ana.truth_children["lep"]
    lep_ch.Color = "red"

    had_ch = TH1F()
    had_ch.Title = "Hadronic"
    had_ch.xData = ana.truth_children["had"]
    had_ch.Color = "aqua"

    all_t = path(TH1F())
    all_t.Histograms = [lep_ch, had_ch]
    all_t.Title = "Top Matching Scheme using Top-Children"
    all_t.xTitle = "Invariant Mass of Matched Top (GeV)"
    all_t.yTitle = "Density (Arb.) / 0.3 GeV"
    all_t.xMin = 140
    all_t.xMax = 200
    all_t.xBins = 240
    all_t.xStep = 5
    all_t.Density = True
    all_t.Overflow = False
    all_t.yLogarithmic = True
    all_t.Filename = "Figure.2.b"
    all_t.SaveFigure()

def top_decay_channel_truth_jets(ana):

    lep_ch = TH1F()
    lep_ch.Title = "Leptonic"
    lep_ch.xData = ana.truth_jets["lep"]
    lep_ch.Color = "red"

    had_ch = TH1F()
    had_ch.Title = "Hadronic"
    had_ch.xData = ana.truth_jets["had"]
    had_ch.Color = "aqua"

    all_t = path(TH1F())
    all_t.Histograms = [lep_ch, had_ch]
    all_t.Title = "Top Matching Scheme using Truth-Jets \n including Truth $\\ell$ and $\\nu$ for Leptonically Decaying Tops"
    all_t.xTitle = "Invariant Mass of Matched Top (GeV)"
    all_t.yTitle = "Density (Arb.) / 1 GeV"
    all_t.xMin = 0
    all_t.xMax = 400
    all_t.xBins = 400
    all_t.xStep = 20
    all_t.Density = True
    all_t.Overflow = False
    all_t.yLogarithmic = True
    all_t.Filename = "Figure.2.c"
    all_t.SaveFigure()

def top_decay_channel_jets_truth_leps(ana):

    lep_ch = TH1F()
    lep_ch.Title = "Leptonic"
    lep_ch.xData = ana.jets_truth_leps["lep"]
    lep_ch.Color = "red"

    had_ch = TH1F()
    had_ch.Title = "Hadronic"
    had_ch.xData = ana.jets_truth_leps["had"]
    had_ch.Color = "aqua"

    all_t = path(TH1F())
    all_t.Histograms = [lep_ch, had_ch]
    all_t.Title = "Top Matching Scheme using Jets \n including Truth $\\ell$ and $\\nu$ for Leptonically Decaying Tops"
    all_t.xTitle = "Invariant Mass of Matched Top (GeV)"
    all_t.yTitle = "Density (Arb.) / 1 GeV"
    all_t.xMin = 0
    all_t.xMax = 400
    all_t.xBins = 400
    all_t.xStep = 20
    all_t.Density = True
    all_t.Overflow = False
    all_t.yLogarithmic = True
    all_t.Filename = "Figure.2.d"
    all_t.SaveFigure()

def top_decay_channel_jets_leps(ana):

    lep_ch = TH1F()
    lep_ch.Title = "Leptonic"
    lep_ch.xData = ana.jet_leps["lep"]
    lep_ch.Color = "red"

    had_ch = TH1F()
    had_ch.Title = "Hadronic"
    had_ch.xData = ana.jet_leps["had"]
    had_ch.Color = "aqua"

    all_t = path(TH1F())
    all_t.Histograms = [lep_ch, had_ch]
    all_t.Title = "Top Matching Scheme using Detector Jets and $\\ell$ \n including Truth $\\nu$ for Leptonically Decaying Tops"
    all_t.xTitle = "Invariant Mass of Matched Top (GeV)"
    all_t.yTitle = "Density (Arb.) / 1 GeV"
    all_t.xMin = 0
    all_t.xMax = 400
    all_t.xBins = 400
    all_t.xStep = 20
    all_t.Density = True
    all_t.Overflow = False
    all_t.yLogarithmic = True
    all_t.Filename = "Figure.2.e"
    all_t.SaveFigure()

    hists = []
    cl = iter(["orange", "green", "magenta", "red"])
    for i in ana.jet_leps:
        if "-" not in i: continue
        kx = i.split("-")[-1]
        if kx == "11": kx = "electron"
        if kx == "13": kx = "muon"
        if kx == "15": kx = "tau"
        if kx == "miss": kx = "missed"
        th = TH1F()
        th.Alpha = 0.5
        th.Title = kx
        th.xData = ana.jet_leps[i]
        th.Color = next(cl)
        hists.append(th)

    all_t = path(TH1F())
    all_t.Histograms = hists
    all_t.Title = "Lepton PDGID Segmentation for leptonically Decaying Tops"
    all_t.xTitle = "Invariant Mass of Matched Top (GeV)"
    all_t.yTitle = "Density (Arb.) / 1 GeV"
    all_t.xMin = 0
    all_t.xMax = 400
    all_t.xBins = 400
    all_t.xStep = 20
    all_t.Density = True
    all_t.Overflow = False
    all_t.Stacked = True
    all_t.yLogarithmic = True
    all_t.Filename = "Figure.2.e_pdgid"
    all_t.SaveFigure()



def top_truth_jets_contributions(ana):
    an_ = sorted(ana.n_truth_jets_lep)
    njets = {k : int(k.split(" ")[0]) for k in an_}
    hsts = {k : TH1F() for k in njets.values() if k <= 4}
    for ntj in an_:
        n = njets[ntj]
        if n >= 4: data = hsts[4]
        else: data = hsts[n]
        data.xData += ana.n_truth_jets_lep[ntj]
        if 1 == n: ntj = ntj.replace("Jets", "Jet")
        if n > 3: ntj = " $\\geq 4$ - Truth Jets"
        data.Title = ntj

    cl = iter(["orange", "green", "magenta", "red"])
    for i in hsts.values(): i.Color = next(cl); i.Alpha = 0.5

    all_t1 = path(TH1F())
    all_t1.Histograms = list(hsts.values())
    all_t1.Title = "n-Truth Jets Matched to Mutual Leptonically Decaying Tops \n (using Truth $\\ell$ + $\\nu$)"
    all_t1.xTitle = "Invariant Mass of Matched Top (GeV)"
    all_t1.yTitle = "Entries / 1 GeV"
    all_t1.xMin = 0
    all_t1.xMax = 400
    all_t1.xBins = 400
    all_t1.xStep = 20
    all_t1.Stacked = True
    all_t1.Density = True
    all_t1.Overflow = False
    all_t1.yLogarithmic = True
    all_t1.Filename = "Figure.2.f"
    all_t1.SaveFigure()

    an_ = sorted(ana.n_truth_jets_had)
    njets = {k : int(k.split(" ")[0]) for k in an_}
    hsts = {k : TH1F() for k in njets.values() if k <= 4}
    for ntj in an_:
        n = njets[ntj]
        if n >= 4: data = hsts[4]
        else: data = hsts[n]
        data.xData += ana.n_truth_jets_had[ntj]
        if 1 == n: ntj = ntj.replace("Jets", "Jet")
        if n > 3: ntj = " $\\geq 4$ - Truth Jets"
        data.Title = ntj

    cl = iter(["orange", "green", "magenta", "red"])
    for i in hsts.values(): i.Color = next(cl); i.Alpha = 0.5

    all_t2 = path(TH1F())
    all_t2.Histograms = list(hsts.values())
    all_t2.Title = "n-Truth Jets Matched to Mutual Hadronically Decaying Tops"
    all_t2.xTitle = "Invariant Mass of Matched Top (GeV)"
    all_t2.yTitle = "Entries / 1 GeV"
    all_t2.xMin = 0
    all_t2.xMax = 400
    all_t2.xBins = 400
    all_t2.xStep = 20
    all_t2.Stacked = True
    all_t2.yLogarithmic = True
    all_t2.Density = True
    all_t2.Overflow = False
    all_t2.Filename = "Figure.2.g"
    all_t2.SaveFigure()

def top_jets_contributions(ana):
    an_ = sorted(ana.n_jets_lep)
    njets = {k : int(k.split(" ")[0]) for k in an_}
    hsts = {k : TH1F() for k in njets.values() if k <= 4}
    for ntj in an_:
        n = njets[ntj]
        if n >= 4: data = hsts[4]
        else: data = hsts[n]
        data.xData += ana.n_jets_lep[ntj]

        if 1 == n: ntj = ntj.replace("Jets", "Jet")
        if n > 3: ntj = " $\\geq 4$ - Jets"

        data.Title = ntj

    cl = iter(["orange", "green", "magenta", "red"])
    for i in hsts.values(): i.Color = next(cl); i.Alpha = 0.5

    all_t1 = path(TH1F())
    all_t1.Histograms = list(hsts.values())
    all_t1.Title = "n-Jets Matched to Mutual Leptonically Decaying Tops \n (using Detector $\\ell$ and Truth $\\nu$)"
    all_t1.xTitle = "Invariant Mass of Matched Top (GeV)"
    all_t1.yTitle = "Entries / 1 GeV"
    all_t1.xMin = 0
    all_t1.xMax = 400
    all_t1.xBins = 400
    all_t1.xStep = 20
    all_t1.Stacked = True
    all_t1.yLogarithmic = True
    all_t1.Density = True
    all_t1.Overflow = False
    all_t1.Filename = "Figure.2.h"
    all_t1.SaveFigure()

    an_ = sorted(ana.n_jets_had)
    njets = {k : int(k.split(" ")[0]) for k in an_}
    hsts = {k : TH1F() for k in njets.values() if k <= 4}
    for ntj in an_:
        n = njets[ntj]
        if n >= 4: data = hsts[4]
        else: data = hsts[n]
        data.xData += ana.n_jets_had[ntj]
        if 1 == n: ntj = ntj.replace("Jets", "Jet")
        if n > 3: ntj = " $\\geq 4$ - Jets"
        data.Title = ntj

    cl = iter(["orange", "green", "magenta", "red"])
    for i in hsts.values(): i.Color = next(cl); i.Alpha = 0.5

    all_t2 = path(TH1F())
    all_t2.Histograms = list(hsts.values())
    all_t2.Title = "n-Jets Matched to Mutual Hadronically Decaying Tops"
    all_t2.xTitle = "Invariant Mass of Matched Top (GeV)"
    all_t2.yTitle = "Entries / 1 GeV"
    all_t2.xMin = 0
    all_t2.xMax = 400
    all_t2.xBins = 400
    all_t2.xStep = 20
    all_t2.Stacked = True
    all_t2.Density = True
    all_t2.Overflow = False
    all_t2.yLogarithmic = True
    all_t2.Filename = "Figure.2.i"
    all_t2.SaveFigure()

def TopMatching(ana):
    top_matching(ana)
    top_decay_channel_children(ana)
    top_decay_channel_truth_jets(ana)
    top_decay_channel_jets_truth_leps(ana)
    top_decay_channel_jets_leps(ana)
    top_truth_jets_contributions(ana)
    top_jets_contributions(ana)

from AnalysisG.core.plotting import TH1F, TH2F

global figure_path
global mass_point

def path(hist):
    hist.OutputDirectory = figure_path + "/top-matching/" + mass_point
    return hist

def top_matching(ana):

    tru_top = TH1F()
    tru_top.Title = "Truth-Top"
    tru_top.xData = ana.truth_top
    tru_top.Density = True

    tru_ch = TH1F()
    tru_ch.Title = "Truth-Children"
    tru_ch.xData = ana.truth_children["all"]
    tru_ch.Density = True

    tru_tj = TH1F()
    tru_tj.Title = "Truth-Jets (Truth Leptons and Neutrinos)"
    tru_tj.xData = ana.truth_jets["all"]
    tru_tj.Density = True

    tru_j = TH1F()
    tru_j.Title = "Jets (Truth Leptons and Neutrinos)"
    tru_j.xData = ana.jets_truth_leps["all"]
    tru_j.Density = True

    tru_jl = TH1F()
    tru_jl.Title = "Jets Leptons (Truth Neutrinos)"
    tru_jl.xData = ana.jet_leps["all"]
    tru_jl.Density = True

    all_t = path(TH1F())
    all_t.Histograms = [tru_jl, tru_j, tru_tj, tru_ch, tru_top]
    all_t.Title = "Top Truth Matching Scheme for Varying Level of Monte Carlo Truth"
    all_t.xTitle = "Invariant Mass (GeV)"
    all_t.yTitle = "Entries (Arb.)"
    all_t.xMin = 0
    all_t.xMax = 400
    all_t.xBins = 1000
    all_t.xStep = 50
    all_t.yLogarithmic = True
    all_t.Density = True
    all_t.Filename = "Figure.2.a"
    all_t.SaveFigure()

def top_decay_channel_children(ana):

    all_ch = TH1F()
    all_ch.Title = "All"
    all_ch.xData = ana.truth_children["all"]

    lep_ch = TH1F()
    lep_ch.Title = "Leptonic"
    lep_ch.xData = ana.truth_children["lep"]

    had_ch = TH1F()
    had_ch.Title = "Hadronic"
    had_ch.xData = ana.truth_children["had"]

    all_t = path(TH1F())
    all_t.Histograms = [all_ch, lep_ch, had_ch]
    all_t.Title = "Top Truth Matching Scheme to Truth-Children for different Decay Modes"
    all_t.xTitle = "Invariant Mass (GeV)"
    all_t.yTitle = "Entries (Arb.)"
    all_t.xMin = 100
    all_t.xMax = 250
    all_t.xBins = 1000
    all_t.xStep = 40
    all_t.yLogarithmic = True
    all_t.Filename = "Figure.2.b"
    all_t.SaveFigure()

def top_decay_channel_truth_jets(ana):

    all_ch = TH1F()
    all_ch.Title = "All"
    all_ch.xData = ana.truth_jets["all"]

    lep_ch = TH1F()
    lep_ch.Title = "Leptonic"
    lep_ch.xData = ana.truth_jets["lep"]

    had_ch = TH1F()
    had_ch.Title = "Hadronic"
    had_ch.xData = ana.truth_jets["had"]

    all_t = path(TH1F())
    all_t.Histograms = [all_ch, lep_ch, had_ch]
    all_t.Title = "Top Truth Matching Scheme to Truth-Jets with \n Truth Leptons and Neutrinos for different Decay Modes"
    all_t.xTitle = "Invariant Mass (GeV)"
    all_t.yTitle = "Entries (Arb.)"
    all_t.xMin = 0
    all_t.xMax = 500
    all_t.xBins = 1000
    all_t.xStep = 40
    all_t.yLogarithmic = True
    all_t.Filename = "Figure.2.c"
    all_t.SaveFigure()

def top_decay_channel_jets_truth_leps(ana):

    all_ch = TH1F()
    all_ch.Title = "All"
    all_ch.xData = ana.jets_truth_leps["all"]

    lep_ch = TH1F()
    lep_ch.Title = "Leptonic"
    lep_ch.xData = ana.jets_truth_leps["lep"]

    had_ch = TH1F()
    had_ch.Title = "Hadronic"
    had_ch.xData = ana.jets_truth_leps["had"]

    all_t = path(TH1F())
    all_t.Histograms = [all_ch, lep_ch, had_ch]
    all_t.Title = "Top Truth Matching Scheme to Jets with \n Truth Leptons and Neutrinos for different Decay Modes"
    all_t.xTitle = "Invariant Mass (GeV)"
    all_t.yTitle = "Entries (Arb.)"
    all_t.xMin = 0
    all_t.xMax = 500
    all_t.xBins = 1000
    all_t.xStep = 40
    all_t.yLogarithmic = True
    all_t.Filename = "Figure.2.d"
    all_t.SaveFigure()

def top_decay_channel_jets_leps(ana):

    all_ch = TH1F()
    all_ch.Title = "All"
    all_ch.xData = ana.jet_leps["all"]

    lep_ch = TH1F()
    lep_ch.Title = "Leptonic"
    lep_ch.xData = ana.jet_leps["lep"]

    had_ch = TH1F()
    had_ch.Title = "Hadronic"
    had_ch.xData = ana.jet_leps["had"]

    all_t = path(TH1F())
    all_t.Histograms = [all_ch, lep_ch, had_ch]
    all_t.Title = "Top Truth Matching Scheme to Jets with Detector Leptons and using \n Truth Neutrinos for different Decay Modes"
    all_t.xTitle = "Invariant Mass (GeV)"
    all_t.yTitle = "Entries (Arb.)"
    all_t.xMin = 0
    all_t.xMax = 500
    all_t.xBins = 1000
    all_t.xStep = 40
    all_t.yLogarithmic = True
    all_t.Filename = "Figure.2.e"
    all_t.SaveFigure()

def top_truth_jets_contributions(ana):

    hists = []
    an_ = sorted(ana.n_truth_jets_lep)
    for ntj in an_:
        data = TH1F()
        data.xData = ana.n_truth_jets_lep[ntj]
        if "1 -" in ntj: ntj = ntj.replace("Jets", "Jet")
        data.Title = ntj
        hists += [data]

    all_t1 = path(TH1F())
    all_t1.Histograms = hists
    all_t1.Title = "Reconstructed Invariant Mass of Truth Tops from Truth Jets and Truth Children \n (Leptonic Decay Mode)"
    all_t1.xTitle = "Invariant Mass (GeV)"
    all_t1.yTitle = "Entries (Arb.)"
    all_t1.xMin = 0
    all_t1.xMax = 500
    all_t1.xBins = 1000
    all_t1.xStep = 40
    all_t1.yLogarithmic = True
    all_t1.Stacked = True
    all_t1.Filename = "Figure.2.f"
    all_t1.SaveFigure()


    hists = []
    an_ = sorted(ana.n_truth_jets_had)
    for ntj in an_:
        data = TH1F()
        data.xData = ana.n_truth_jets_had[ntj]
        if "1 -" in ntj: ntj = ntj.replace("Jets", "Jet")
        data.Title = ntj
        hists += [data]

    all_t2 = path(TH1F())
    all_t2.Histograms = hists
    all_t2.Title = "Reconstructed Invariant Mass of Truth Tops from Truth Jets \n (Hadronic Decay Mode)"
    all_t2.xTitle = "Invariant Mass (GeV)"
    all_t2.yTitle = "Entries (Arb.)"
    all_t2.xMin = 0
    all_t2.xMax = 500
    all_t2.xBins = 1000
    all_t2.xStep = 40
    all_t2.Stacked = True
    all_t2.yLogarithmic = True
    all_t2.Filename = "Figure.2.g"
    all_t2.SaveFigure()

    xdata = {i : [] for i in list(ana.n_truth_jets_had) + list(ana.n_truth_jets_lep)}
    for ntj in sorted(xdata):
        try: xdata[ntj] += ana.n_truth_jets_lep[ntj]
        except KeyError: pass
        try: xdata[ntj] += ana.n_truth_jets_had[ntj]
        except KeyError: pass

    ydata = sum([[int(i.split(" ")[0])]*len(xdata[i]) for i in xdata], [])
    xdata = sum(xdata.values(), [])

    th = path(TH2F())
    th.Title = "Reconstructed Invariant Top-Quark Mass as a Function of Number of Truth Jets \n (Combined hadronic and leptonic modes)"
    th.xData = ydata
    th.xMin = 0
    th.xMax = 9
    th.xBins = 9
    th.xStep = 1
    th.xTitle = "Number of Truth Jet Contributions"

    th.yData = xdata
    th.yMin = 100
    th.yMax = 250
    th.yBins = 400
    th.yStep = 20

    th.yTitle = "Invariant Top-Quark Mass (GeV)"
    th.Filename = "Figure.2.h"
    th.SaveFigure()


def top_jets_contributions(ana):

    hists = []
    an_ = sorted(ana.n_jets_lep)
    for ntj in an_:
        data = TH1F()
        data.xData = ana.n_jets_lep[ntj]
        if "1 -" in ntj: ntj = ntj.replace("Jets", "Jet")
        data.Title = ntj
        hists += [data]

    all_t1 = path(TH1F())
    all_t1.Histograms = hists
    all_t1.Title = "Reconstructed Invariant Mass of Truth Tops from Jets and Detector Leptons \n with Truth Neutrino (Leptonic Decay Mode)"
    all_t1.xTitle = "Invariant Mass (GeV)"
    all_t1.yTitle = "Entries (Arb.)"
    all_t1.xMin = 0
    all_t1.xMax = 500
    all_t1.xBins = 1000
    all_t1.xStep = 40
    all_t1.Stacked = True
    all_t1.yLogarithmic = True
    all_t1.Filename = "Figure.2.i"
    all_t1.SaveFigure()


    hists = []
    an_ = sorted(ana.n_truth_jets_had)
    for ntj in an_:
        data = TH1F()
        data.xData = ana.n_truth_jets_had[ntj]
        if "1 -" in ntj: ntj = ntj.replace("Jets", "Jet")
        data.Title = ntj
        hists += [data]

    all_t2 = path(TH1F())
    all_t2.Histograms = hists
    all_t2.Title = "Reconstructed Invariant Mass of Truth Tops from Jets (Hadronic)"
    all_t2.xTitle = "Invariant Mass (GeV)"
    all_t2.yTitle = "Entries (Arb.)"
    all_t2.xMin = 0
    all_t2.xMax = 500
    all_t2.xBins = 1000
    all_t2.xStep = 40
    all_t2.Stacked = True
    all_t2.yLogarithmic = True
    all_t2.Filename = "Figure.2.j"
    all_t2.SaveFigure()

    xdata = {i : [] for i in list(ana.n_truth_jets_had) + list(ana.n_jets_lep)}
    for ntj in sorted(xdata):
        try: xdata[ntj] += ana.n_jets_lep[ntj]
        except KeyError: pass
        try: xdata[ntj] += ana.n_jets_had[ntj]
        except KeyError: pass

    ydata = sum([[int(i.split(" ")[0])]*len(xdata[i]) for i in xdata], [])
    xdata = sum(xdata.values(), [])

    th = path(TH2F())
    th.Title = "Reconstructed Invariant Top-Quark Mass as a Function of Number of Jet Constributions \n (Combined hadronic and leptonic modes)"

    th.xData = ydata
    th.xMin = 0
    th.xMax = 9
    th.xBins = 9
    th.xStep = 1
    th.xTitle = "Number of Jet Contributions"

    th.yData = xdata
    th.yMin = 100
    th.yMax = 250
    th.yBins = 400
    th.yStep = 20

    th.yTitle = "Invariant Top-Quark Mass (GeV)"
    th.Filename = "Figure.2.k"
    th.SaveFigure()

def TopMatching(ana):
    top_matching(ana)
    top_decay_channel_children(ana)
    top_decay_channel_truth_jets(ana)
    top_decay_channel_jets_truth_leps(ana)
    top_decay_channel_jets_leps(ana)
    top_truth_jets_contributions(ana)
    top_jets_contributions(ana)

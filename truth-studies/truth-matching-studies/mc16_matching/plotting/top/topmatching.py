from AnalysisG.Plotting import TH1F, TH2F

def settings():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : "./plt_plots/top/",
            "Histograms" : [],
            "Histogram" : None,
            "LegendLoc" : "upper right"
    }
    return settings

def top_matching(ana):

    tru_top = TH1F()
    tru_top.Title = "Truth-Top"
    tru_top.xData = ana.truth_top

    tru_ch = TH1F()
    tru_ch.Title = "Truth-Children"
    tru_ch.xData = ana.truth_children["all"]

    tru_tj = TH1F()
    tru_tj.Title = "Truth-Jets (Truth Leptons and Neutrinos)"
    tru_tj.xData = ana.truth_jets["all"]

    tru_j = TH1F()
    tru_j.Title = "Jets (Truth Leptons and Neutrinos)"
    tru_j.xData = ana.jets_truth_leps["all"]

    tru_jl = TH1F()
    tru_jl.Title = "Jets Leptons (Truth Neutrinos)"
    tru_jl.xData = ana.jets_leps["all"]

    sett = settings()
    all_t = TH1F(**sett)
    all_t.Histograms = [tru_jl, tru_j, tru_tj, tru_ch, tru_top]
    all_t.Title = "Top Truth Matching Scheme for Varying Level of Monte Carlo Truth"
    all_t.xTitle = "Invariant Mass (GeV)"
    all_t.yTitle = "Entries <unit>"
    all_t.xMin = 0
    all_t.xMax = 400
    all_t.xBins = 400
    all_t.xStep = 20
    all_t.yScaling = 10
    all_t.xScaling = 20
    all_t.FontSize = 20
    all_t.LabelSize = 20
    all_t.OverFlow = True
    all_t.yLogarithmic = True
    all_t.Filename = "Figure.4.a"
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

    sett = settings()
    all_t = TH1F(**sett)
    all_t.Histograms = [all_ch, lep_ch, had_ch]
    all_t.Title = "Top Truth Matching Scheme to Truth-Children for different Decay Modes"
    all_t.xTitle = "Invariant Mass (GeV)"
    all_t.yTitle = "Entries <unit>"
    all_t.xMin = 0
    all_t.xMax = 400
    all_t.xBins = 400
    all_t.xStep = 40
    all_t.OverFlow = True
    all_t.yLogarithmic = True
    all_t.Filename = "Figure.4.b"
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

    sett = settings()
    all_t = TH1F(**sett)
    all_t.Histograms = [all_ch, lep_ch, had_ch]
    all_t.Title = "Top Truth Matching Scheme to Truth-Jets with \n Truth Leptons and Neutrinos for different Decay Modes"
    all_t.xTitle = "Invariant Mass (GeV)"
    all_t.yTitle = "Entries <unit>"
    all_t.xMin = 0
    all_t.xMax = 400
    all_t.xBins = 400
    all_t.xStep = 40
    all_t.OverFlow = True
    all_t.yLogarithmic = True
    all_t.Filename = "Figure.4.c"
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

    sett = settings()
    all_t = TH1F(**sett)
    all_t.Histograms = [all_ch, lep_ch, had_ch]
    all_t.Title = "Top Truth Matching Scheme to Jets with \n Truth Leptons and Neutrinos for different Decay Modes"
    all_t.xTitle = "Invariant Mass (GeV)"
    all_t.yTitle = "Entries <unit>"
    all_t.xMin = 0
    all_t.xMax = 400
    all_t.xBins = 400
    all_t.xStep = 40
    all_t.OverFlow = True
    all_t.yLogarithmic = True
    all_t.Filename = "Figure.4.d"
    all_t.SaveFigure()

def top_decay_channel_jets_leps(ana):

    all_ch = TH1F()
    all_ch.Title = "All"
    all_ch.xData = ana.jets_leps["all"]

    lep_ch = TH1F()
    lep_ch.Title = "Leptonic"
    lep_ch.xData = ana.jets_leps["lep"]

    had_ch = TH1F()
    had_ch.Title = "Hadronic"
    had_ch.xData = ana.jets_leps["had"]

    sett = settings()
    all_t = TH1F(**sett)
    all_t.Histograms = [all_ch, lep_ch, had_ch]
    all_t.Title = "Top Truth Matching Scheme to Jets with Detector Leptons and using \n Truth Neutrinos for different Decay Modes"
    all_t.xTitle = "Invariant Mass (GeV)"
    all_t.yTitle = "Entries <unit>"
    all_t.xMin = 0
    all_t.xMax = 400
    all_t.xBins = 400
    all_t.xStep = 40
    all_t.OverFlow = True
    all_t.yLogarithmic = True
    all_t.Filename = "Figure.4.e"
    all_t.SaveFigure()

def top_truth_jets_contributions(ana):

    hists = []
    an_ = sorted(ana.n_truth_jets_lep)
    for ntj in an_:
        data = TH1F()
        data.Title = ntj
        data.xData = ana.n_truth_jets_lep[ntj]
        hists += [data]

    sett = settings()
    all_t = TH1F(**sett)
    all_t.Histograms = hists
    all_t.Title = "Reconstructed Invariant Mass of Truth Tops from Truth Jets and Truth Children (Leptonic Decay Mode)"
    all_t.xTitle = "Invariant Mass (GeV)"
    all_t.yTitle = "Entries <unit>"
    all_t.xMin = 0
    all_t.xMax = 400
    all_t.xBins = 400
    all_t.xStep = 20
    all_t.yScaling = 10
    all_t.xScaling = 20
    all_t.FontSize = 20
    all_t.LabelSize = 20
    all_t.OverFlow = True
    all_t.yLogarithmic = True
    all_t.Filename = "Figure.4.f"
    all_t.SaveFigure()


    hists = []
    an_ = sorted(ana.n_truth_jets_had)
    for ntj in an_:
        data = TH1F()
        data.Title = ntj
        data.xData = ana.n_truth_jets_had[ntj]
        hists += [data]

    sett = settings()
    all_t = TH1F(**sett)
    all_t.Histograms = hists
    all_t.Title = "Reconstructed Invariant Mass of Truth Tops from Truth Jets (Hadronic)"
    all_t.xTitle = "Invariant Mass (GeV)"
    all_t.yTitle = "Entries <unit>"
    all_t.xMin = 0
    all_t.xMax = 400
    all_t.xBins = 400
    all_t.xStep = 20
    all_t.yScaling = 10
    all_t.xScaling = 20
    all_t.FontSize = 20
    all_t.LabelSize = 20
    all_t.OverFlow = True
    all_t.yLogarithmic = True
    all_t.Filename = "Figure.4.g"
    all_t.SaveFigure()

    xdata = {i : [] for i in list(ana.n_truth_jets_had) + list(ana.n_truth_jets_lep)}
    for ntj in sorted(xdata):
        try: xdata[ntj] += ana.n_truth_jets_lep[ntj]
        except KeyError: pass
        try: xdata[ntj] += ana.n_truth_jets_had[ntj]
        except KeyError: pass

    ydata = sum([[int(i.split(" ")[0])]*len(xdata[i]) for i in xdata], [])
    xdata = sum(xdata.values(), [])

    th = TH2F()
    th.OutputDirectory = "./plt_plots/top/"
    th.Filename = "Figure.4.h"
    th.Title = "Reconstructed Invariant Top-Quark Mass as a Function of Number of Truth Jet Constributions \n (Combined hadronic and leptonic modes)"

    th.xData = ydata
    th.xMin = 0
    th.xMax = 9
    th.xBins = 9
    th.xStep = 1
    th.xOverFlow = True
    th.xTitle = "Number of Truth Jet Contributions"

    th.yData = xdata
    th.yMin = 100
    th.yMax = 250
    th.yBins = 400
    th.yStep = 20

    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.Color = "YlOrBr"
    th.yOverFlow = True
    th.yTitle = "Invariant Top-Quark Mass (GeV)"
    th.SaveFigure()


def top_jets_contributions(ana):

    hists = []
    an_ = sorted(ana.n_jets_lep)
    for ntj in an_:
        data = TH1F()
        data.Title = ntj
        data.xData = ana.n_jets_lep[ntj]
        hists += [data]

    sett = settings()
    all_t = TH1F(**sett)
    all_t.Histograms = hists
    all_t.Title = "Reconstructed Invariant Mass of Truth Tops from Jets and Detector Leptons with Truth Neutrino (Leptonic Decay Mode)"
    all_t.xTitle = "Invariant Mass (GeV)"
    all_t.yTitle = "Entries <unit>"
    all_t.xMin = 0
    all_t.xMax = 400
    all_t.xBins = 400
    all_t.xStep = 20
    all_t.yScaling = 10
    all_t.xScaling = 20
    all_t.FontSize = 20
    all_t.LabelSize = 20
    all_t.OverFlow = True
    all_t.yLogarithmic = True
    all_t.Filename = "Figure.4.i"
    all_t.SaveFigure()


    hists = []
    an_ = sorted(ana.n_truth_jets_had)
    for ntj in an_:
        data = TH1F()
        data.Title = ntj
        data.xData = ana.n_truth_jets_had[ntj]
        hists += [data]

    sett = settings()
    all_t = TH1F(**sett)
    all_t.Histograms = hists
    all_t.Title = "Reconstructed Invariant Mass of Truth Tops from Jets (Hadronic)"
    all_t.xTitle = "Invariant Mass (GeV)"
    all_t.yTitle = "Entries <unit>"
    all_t.xMin = 0
    all_t.xMax = 400
    all_t.xBins = 400
    all_t.xStep = 20
    all_t.yScaling = 10
    all_t.xScaling = 20
    all_t.FontSize = 20
    all_t.LabelSize = 20
    all_t.OverFlow = True
    all_t.yLogarithmic = True
    all_t.Filename = "Figure.4.j"
    all_t.SaveFigure()

    xdata = {i : [] for i in list(ana.n_truth_jets_had) + list(ana.n_jets_lep)}
    for ntj in sorted(xdata):
        try: xdata[ntj] += ana.n_jets_lep[ntj]
        except KeyError: pass
        try: xdata[ntj] += ana.n_jets_had[ntj]
        except KeyError: pass

    ydata = sum([[int(i.split(" ")[0])]*len(xdata[i]) for i in xdata], [])
    xdata = sum(xdata.values(), [])

    th = TH2F()
    th.OutputDirectory = "./plt_plots/top/"
    th.Filename = "Figure.4.k"
    th.Title = "Reconstructed Invariant Top-Quark Mass as a Function of Number of Jet Constributions \n (Combined hadronic and leptonic modes)"

    th.xData = ydata
    th.xMin = 0
    th.xMax = 9
    th.xBins = 9
    th.xStep = 1
    th.xOverFlow = True
    th.xTitle = "Number of Jet Contributions"

    th.yData = xdata
    th.yMin = 100
    th.yMax = 250
    th.yBins = 400
    th.yStep = 20

    th.yScaling = 10
    th.xScaling = 20
    th.FontSize = 20
    th.LabelSize = 20
    th.Color = "YlOrBr"
    th.yOverFlow = True
    th.yTitle = "Invariant Top-Quark Mass (GeV)"
    th.SaveFigure()

def TopMatching(ana):
    top_matching(ana)
    top_decay_channel_children(ana)
    top_decay_channel_truth_jets(ana)
    top_decay_channel_jets_truth_leps(ana)
    top_decay_channel_jets_leps(ana)
    top_truth_jets_contributions(ana)
    top_jets_contributions(ana)

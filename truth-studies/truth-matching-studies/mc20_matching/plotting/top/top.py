from AnalysisG.Plotting import TH1F, TH2F
global figure_path

def settings():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : figure_path + "top/figures/",
            "Histograms" : [],
            "Histogram" : None,
            "FontSize" : 15,
            "LabelSize" : 20,
            "xScaling" : 10,
            "yScaling" : 12,
            "LegendLoc" : "upper right"
    }
    return settings

def settings_th2f():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : figure_path + "top/figures/",
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
    all_t.Filename = "Figure.1.a"
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
    all_t.Filename = "Figure.1.b"
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
    all_t.Filename = "Figure.1.c"
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
#    all_t.OverFlow = True
#    all_t.yLogarithmic = True
    all_t.Filename = "Figure.1.d"
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
#    all_t.OverFlow = True
#    all_t.yLogarithmic = True
    all_t.Filename = "Figure.1.e"
    all_t.SaveFigure()



def TopMatching(ana):
    top_matching(ana)
    top_decay_channel_children(ana)
    top_decay_channel_truth_jets(ana)
    top_decay_channel_jets_truth_leps(ana)
    top_decay_channel_jets_leps(ana)

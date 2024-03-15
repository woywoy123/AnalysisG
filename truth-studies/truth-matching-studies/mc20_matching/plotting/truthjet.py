from AnalysisG.Plotting import TH1F, TH2F

def settings():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : "./plt_plots/truth-jets",
            "Histograms" : [],
            "Histogram" : None,
            "LegendLoc" : "upper right"
    }
    return {i : x for i, x in settings.items()}

def plot_mass(ana):
    data = ana.abstract_top
    sett = settings()
    th = TH1F(**sett)
    for i in data:
        th1 = TH1F()
        th1.xData = data[i]
        th1.Title = "top-index-" + str(i)
        th.Histograms += [th1]

    th.Title = "Invariant Mass of Truth-Top Matched to Truth-Particles (Truth-Jets/Neutrinos/Leptons)"
    th.xMin = 150
    th.xMax = 230
    th.xBins = 250
    th.xStep = 10
    th.yTitle = "Entries"
    th.xTitle = "Invariant Mass (GeV)"
    th.Filename = "Figure.2.a"
    th.Stack = True
    th.SaveFigure()

    hist = []
    data = ana.decaymode
    th = TH1F(**sett)
    for i in data:
        th1 = TH1F()
        th1.xData = data[i]
        th1.Title = i
        hist.append(th1)

    th.Title = "Invariant Mass of Truth-Tops Segmented into Decay Mode"
    th.Histograms = hist
    th.xMin = 0
    th.xMax = 500
    th.xBins = 500
    th.xStep = 100
    th.yTitle = "Entries"
    th.xTitle = "Invariant Mass (GeV)"
    th.Filename = "Figure.2.b"
    th.Stack = True
    th.SaveFigure()





def TruthJetMatching(ana):
    plot_mass(ana)

from AnalysisG.Plotting import TH1F, TH2F
global figure_path

def settings():
    settings = {
            "Style" : "ROOT",
            "OutputDirectory" : figure_path + "truth-jets/figures/",
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
            "OutputDirectory" : figure_path + "truth-jets/figures/",
    }
    return settings

def top_mass_truthjets(ana):

    sett = settings()
    for x in ["leptonic", "hadronic"]:
        th = TH1F()
        th.Title = x
        th.xData = ana.top_mass[x]
        sett["Histograms"] += [th]

    th_ = TH1F(**sett)
    th_.Title = "Invariant Top-Quark Mass from Matched Truth Jets (and Truth Leptons/Neutrinos)"
    th_.xTitle = "Invariant Mass (GeV)"
    th_.yTitle = "Entries <unit>"
    th_.Filename = "Figure.7.a"
    th_.xBins = 500
    th_.xMin  = 0
    th_.xMax  = 500
    th_.xStep = 50
    th_.SaveFigure()


    for x, name in zip(["leptonic", "hadronic"], ["b", "c"]):
        sett = settings()
        for nj in ana.top_mass["ntruthjets"][x]:
            th = TH1F()
            th.Title = str(nj) + "-TruthJets"
            th.xData = ana.top_mass["ntruthjets"][x][nj]
            sett["Histograms"] += [th]

        th_ = TH1F(**sett)
        th_.Title = "Invariant Top-Quark Mass for " + x + " Decays for n-TruthJet Contributions"
        th_.xTitle = "Invariant Mass (GeV)"
        th_.yTitle = "Entries <unit>"
        th_.Filename = "Figure.7." + name
        th_.xBins = 500
        th_.xMin  = 0
        th_.xMax  = 500
        th_.xStep = 50
        th_.Stack = True
        th_.SaveFigure()


    for x, name in zip(["leptonic", "hadronic"], ["d", "e"]):
        sett = settings()
        for nj in ana.top_mass["merged_tops"][x]:
            th = TH1F()
            th.Title = str(nj) + "-Tops"
            th.xData = ana.top_mass["merged_tops"][x][nj]
            sett["Histograms"] += [th]

        th_ = TH1F(**sett)
        th_.Title = "Invariant Top-Quark Mass for " + x + " Decays with \n n-Top Contributions in Truth Jets"
        th_.xTitle = "Invariant Mass (GeV)"
        th_.yTitle = "Entries <unit>"
        th_.Filename = "Figure.7." + name
        th_.xBins = 500
        th_.xMin  = 0
        th_.xMax  = 500
        th_.xStep = 50
        th_.Stack = True
        th_.SaveFigure()

def TopTruthJets(ana):
    top_mass_truthjets(ana)

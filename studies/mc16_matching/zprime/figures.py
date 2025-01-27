from AnalysisG.core.plotting import TH1F, TH2F
import pickle

global figure_path
global mass_point

colors = ["red", "green", "blue", "orange", "magenta", "cyan", "pink"]
def path(hist):
    hist.OutputDirectory = figure_path + "/zprime/" + mass_point
    return hist

def zprime_mass_tops(ana):
    xp = iter(colors)

    hists = []
    for i in ana:
        th_ = TH1F()
        th_.Title = "mass-point " + i
        th_.xData = ana[i].zprime_truth_tops
        th_.Color = next(xp)
        hists.append(th_)

    th = path(TH1F())
    th.Histograms = hists
    th.Title = "Invariant Mass Distribution of Multiple Target Resonance Mass-Points"
    th.xTitle = "Invariant Mass (GeV)"
    th.yTitle = "Density (Arb.) / ($4$ GeV)"
    th.Style = "ATLAS"
    th.xStep = 100
    th.xBins = 300
    th.xMin = 0
    th.xMax = 1200
    th.Density = True
    th.Overflow = False
    th.Filename = "Figure.6.a"
    th.SaveFigure()

def zprime_mass_children(ana):
    xp = iter(colors)

    hists = []
    for i in ana:
        th_ = TH1F()
        th_.Title = "mass-point " + i
        th_.xData = ana[i].zprime_children
        th_.Color = next(xp)
        hists.append(th_)

    th = path(TH1F())
    th.Histograms = hists
    th.Title = "Invariant Mass Distribution of Multiple Target Resonance Mass-Points"
    th.xTitle = "Invariant Mass (GeV)"
    th.yTitle = "Density (Arb.) / ($4$ GeV)"
    th.Style = "ATLAS"
    th.xStep = 100
    th.xBins = 300
    th.xMin = 0
    th.xMax = 1200
    th.Density = True
    th.Overflow = False
    th.Filename = "Figure.6.b"
    th.SaveFigure()

def zprime_mass_truthjets(ana):
    xp = iter(colors)

    hists = []
    for i in ana:
        th_ = TH1F()
        th_.Title = "mass-point " + i
        th_.xData = ana[i].zprime_truthjets
        th_.Color = next(xp)
        hists.append(th_)

    th = path(TH1F())
    th.Histograms = hists
    th.Title = "Invariant Mass Distribution of Multiple Target Resonance Mass-Points"
    th.xTitle = "Invariant Mass (GeV)"
    th.yTitle = "Density (Arb.) / ($4$ GeV)"
    th.Style = "ATLAS"
    th.xStep = 100
    th.xBins = 300
    th.xMin = 0
    th.xMax = 1200
    th.Density = True
    th.Overflow = False
    th.Filename = "Figure.6.c"
    th.SaveFigure()

def zprime_mass_jets(ana):
    xp = iter(colors)

    hists = []
    for i in ana:
        th_ = TH1F()
        th_.Title = "mass-point " + i
        th_.xData = ana[i].zprime_jets
        th_.Color = next(xp)
        hists.append(th_)

    th = path(TH1F())
    th.Histograms = hists
    th.Title = "Invariant Mass Distribution of Multiple Target Resonance Mass-Points"
    th.xTitle = "Invariant Mass (GeV)"
    th.yTitle = "Density (Arb.) / ($4$ GeV)"
    th.Style = "ATLAS"
    th.xStep = 100
    th.xBins = 300
    th.xMin = 0
    th.xMax = 1200
    th.Density = True
    th.Overflow = False
    th.Filename = "Figure.6.d"
    th.SaveFigure()

def ZPrime(ana):

    data_ = {}
    masses = ["1000", "900", "800", "700", "600", "500", "400"]
    for i in masses:
        try: f = open("zprime-Mass." + i + ".GeV.pkl", "rb")
        except: return

        data_[i] = pickle.load(f)
        f.close()
    zprime_mass_tops(data_)
    zprime_mass_children(data_)
    zprime_mass_truthjets(data_)
    zprime_mass_jets(data_)

from AnalysisG.core.plotting import TH1F, TH2F
from AnalysisG.core.io import IO
import pickle

global figure_path
global mass_point
global default
global study

colors = ["red", "green", "blue", "orange", "magenta", "cyan", "pink"]
def path(hist):
    hist.OutputDirectory = figure_path + "/zprime/" + mass_point
    default(hist)
    return hist

def zprime_mass_tops(ana):
    xp = iter(colors)

    hists = []
    for i in ana:
        th_ = TH1F()
        th_.Title = "Mass Point " + i
        th_.xData = ana[i].zprime_truth_tops
        th_.Color = next(xp)
        hists.append(th_)

    th = path(TH1F())
    th.Histograms = hists
    th.Title = r"Invariant Mass Distribution of Multiple Target Resonance Mass-Points"
    th.xTitle = r"Invariant Mass (GeV)"
    th.yTitle = r"Density (Arb.) / ($4$ GeV)"
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
        th_.Title = "Mass Point " + i
        th_.xData = ana[i].zprime_children
        th_.Color = next(xp)
        hists.append(th_)

    th = path(TH1F())
    th.Histograms = hists
    th.Title = "Invariant Mass Distribution of Multiple Target Resonance Mass-Points"
    th.xTitle = "Invariant Mass (GeV)"
    th.yTitle = r"Density (Arb.) / ($4$ GeV)"
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
        th_.Title = "Mass Point " + i
        th_.xData = ana[i].zprime_truthjets
        th_.Color = next(xp)
        hists.append(th_)

    th = path(TH1F())
    th.Histograms = hists
    th.Title = r"Invariant Mass Distribution of Multiple Target Resonance Mass-Points"
    th.xTitle = r"Invariant Mass (GeV)"
    th.yTitle = r"Density (Arb.) / ($4$ GeV)"
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
        th_.Title = "Mass Point " + i
        th_.xData = ana[i].zprime_jets
        th_.Color = next(xp)
        hists.append(th_)

    th = path(TH1F())
    th.Histograms = hists
    th.Title  = r"Invariant Mass Distribution of Multiple Target Resonance Mass-Points"
    th.xTitle = "Invariant Mass (GeV)"
    th.yTitle = r"Density (Arb.) / ($4$ GeV)"
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
    zprime_mass_tops(ana)
    zprime_mass_children(ana)
    zprime_mass_truthjets(ana)
    zprime_mass_jets(ana)

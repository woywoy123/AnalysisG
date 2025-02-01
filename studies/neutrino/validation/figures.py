from AnalysisG.core.plotting import TH1F, TH2F, TLine
from .common import *
from .helper_fig import *

def path(hist):
    hist.UseLateX = False
    hist.Style = "ATLAS"
    hist.OutputDirectory = "Figures/neutrino-validation/"
    return hist

def template_nunu_top_mass(dt, a, b, c, d, mode):
    r1_cu, r2_cu = dt["r1_cu"], dt["r2_cu"]
    r1_rf, r2_rf = dt["r1_rf"], dt["r2_rf"]
    truth_nux = dt["truth_nux"]

    for i in [("n1", a), ("n2", b)]:
        n, f = i
        th1t = template_hist("Truth"              , truth_nux["tmass"][n], "red")
        th1c = template_hist("CUDA - Dynamic"     , r1_cu["tmass"][n]    , "blue")
        th1r = template_hist("Reference - Dynamic", r1_rf["tmass"][n]    , "green")

        th = path(TH1F())
        th.Title = "Invariant Top Quark Mass derived from Neutrino " + n.replace("n", "") + ": " + mode
        th.Histograms = [th1t, th1c, th1r]
        th.xBins = 400
        th.xMax = 400
        th.xMin = 0
        th.xStep = 40
        th.Overflow = False
        th.Density = True
        th.xTitle = "Invariant Top Mass (GeV)"
        th.yTitle = "Density (Arb.) / 1 GeV"
        th.Filename = "Figure.10." + f
        th.SaveFigure()

    for i in [("n1", c), ("n2", d)]:
        n, f = i
        th1t = template_hist("Truth"             , truth_nux["tmass"][n], "red")
        th1c = template_hist("CUDA - Static"     , r2_cu["tmass"][n]    , "blue")
        th1r = template_hist("Reference - Static", r2_rf["tmass"][n]    , "green")

        th = path(TH1F())
        th.Title = "Invariant Top Quark Mass derived from Neutrino " + n.replace("n", "") + ": " + mode
        th.Histograms = [th1t, th1c, th1r]
        th.xBins = 400
        th.xMax = 400
        th.xMin = 0
        th.xStep = 40
        th.Overflow = False
        th.Density = True
        th.xTitle = "Invariant Top Mass (GeV)"
        th.yTitle = "Density (Arb.) / 1 GeV"
        th.Filename = "Figure.10." + f
        th.SaveFigure()

def topchildren_nunu(ana):
    dt = topchildren_nunu_build(ana)
    template_nunu_top_mass(dt, "a.1", "b.1", "c.1", "d.1", " \n Truth Children")

def toptruthjets_nunu(ana):
    dt = toptruthjets_nunu_build(ana)
    template_nunu_top_mass(dt, "a.2", "b.2", "c.2", "d.2", " \n Truth Jets with Leptonic Truth Children")

def topjetchild_nunu(ana):
    dt = topjetchild_nunu_build(ana)
    template_nunu_top_mass(dt, "a.3", "b.3", "c.3", "d.3", " \n Detector Jets with Leptonic Truth Children")

def topdetector_nunu(ana):
    dt = topdetector_nunu_build(ana)
    template_nunu_top_mass(dt, "a.4", "b.4", "c.4", "d.4", " \n Detector Jets and Leptons")




def nunuValidation(ana):
#    topchildren_nunu(ana)
#    toptruthjets_nunu(ana)
#    topjetchild_nunu(ana)
#    topdetector_nunu(ana)
#    LossStatistics()
    pass

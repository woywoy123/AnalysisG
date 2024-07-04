global figure_path

from AnalysisG.core.plotting import TH1F, TH2F

def top_pt(ana):

    thr = TH1F()
    thr.Title = "Resonance"
    thr.xData = ana["res_top_kinematics"]["pt"]

    ths = TH1F()
    ths.Title = "Spectator"
    ths.xData = ana["spec_top_kinematics"]["pt"]

    sett = settings()
    tha = TH1F(**sett)
    tha.Histograms = [thr, ths]
    tha.Title = "Transverse Momenta of Truth Tops"
    tha.xTitle = "Transverse Momenta (GeV)"
    tha.yTitle = "Entries <unit>"
    tha.xMin = 0
    tha.xMax = 1500
    tha.xBins = 100
    #tha.xStep = 150
    tha.Filename = "Figure.3.a"
    tha.SaveFigure()

def TopKinematics(inpt):
    top_pt(inpt)




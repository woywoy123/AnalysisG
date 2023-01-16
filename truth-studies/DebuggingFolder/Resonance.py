from AnalysisTopGNN import Analysis
from AnalysisTopGNN.Events import Event
from AnalysisTopGNN.Plotting import TH1F

direc = "/home/tnom6927/Downloads/Samples/ttH_tttt_m1000/"

Ana = Analysis()
Ana.InputSample("bsm4", direc)
Ana.Event = Event 
Ana.EventCache = True
Ana.DumpPickle = False
Ana.Threads = 8
Ana.Launch()


res = []
tops = []
for i in Ana:
    event = i.Trees["nominal"]
    t = event.Tops
    sig = sum([k for k in t if k.FromRes == 1])
    res.append(sig.CalculateMass())
    tops += [ k.CalculateMass() for k in t ]



Plot = {}
Plot["xData"] = res
Plot["xMin"] = 0
Plot["xBins"] = 1500
Plot["xMax"] = 1500
Plot["xScaling"] = 4
Plot["xTitle"] = "Mass (GeV)"
Plot["Title"] = "Mass of Scalar H Resonance (1000 GeV)"
Plot["Filename"] = "ResonanceFromTops"
x = TH1F(**Plot)
x.SaveFigure()

Plot = {}
Plot["xData"] = tops
Plot["xMin"] = 100
Plot["xMax"] = 300
Plot["xBins"] = 200
Plot["xScaling"] = 4
Plot["xTitle"] = "Mass (GeV)"
Plot["Title"] = "Mass of Truth Tops"
Plot["Filename"] = "TruthTops"
x = TH1F(**Plot)
x.SaveFigure()

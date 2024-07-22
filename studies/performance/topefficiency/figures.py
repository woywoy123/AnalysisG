from AnalysisG.core.plotting import TH1F, TH2F

global figure_path
global mass_point

def path(hist):
    hist.OutputDirectory = figure_path + "/topefficiency/" + mass_point
    return hist

def ChildrenRegions(ana):
    data = ana.truthchildren_pt_eta_topmass
    for i in data:
        th = path(TH1F())
        th.OutputDirectory = th.OutputDirectory + "/1.a/"
        th.Filename = i.replace(" < $pt_{top}$ < ", "->").replace(" < $eta_{top}$ < ", "->").replace("|", ",")
        th.Title = "Invariant Top Mass for Kinematic Regions: " + i.replace("eta", "\eta").replace("|", "\n")
        th.xTitle = "Invariant Top Mass (GeV)"
        th.yTitle = "Entries (Arb.)"
        th.xBins = 500
        th.xMin = 0
        th.xMax = 300
        th.xStep = 20
        th.xData = data[i]
        th.SaveFigure()


def TruthJetsRegions(ana):
    data = ana.truthjets_pt_eta_topmass
    for i in data:
        th = path(TH1F())
        th.OutputDirectory = th.OutputDirectory + "/1.b/"
        th.Filename = i.replace(" < $pt_{top}$ < ", "->").replace(" < $eta_{top}$ < ", "->").replace("|", ",")
        th.Title = "Invariant Top Mass for Kinematic Regions: " + i.replace("eta", "\eta").replace("|", "\n")
        th.xTitle = "Invariant Top Mass (GeV)"
        th.yTitle = "Entries (Arb.)"
        th.xBins = 500
        th.xMin = 0
        th.xMax = 300
        th.xStep = 20
        th.xData = data[i]
        th.SaveFigure()

def JetsRegions(ana):
    data = ana.jets_pt_eta_topmass
    for i in data:
        th = path(TH1F())
        th.OutputDirectory = th.OutputDirectory + "/1.c/"
        th.Filename = i.replace(" < $pt_{top}$ < ", "-").replace(" < $eta_{top}$ < ", "-").replace("|", ",")
        th.Title = "Invariant Top Mass for Kinematic Regions: " + i.replace("eta", "\eta").replace("|", "\n")
        th.xTitle = "Invariant Top Mass (GeV)"
        th.yTitle = "Entries (Arb.)"
        th.xBins = 500
        th.xMin = 0
        th.xMax = 300
        th.xStep = 20
        th.xData = data[i]
        th.SaveFigure()

def TopEfficiency(ana):
    ChildrenRegions(ana)
    TruthJetsRegions(ana)
    JetsRegions(ana)

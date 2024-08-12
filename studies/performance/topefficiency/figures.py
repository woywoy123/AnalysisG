from AnalysisG.core.plotting import TH1F, TH2F
from pathlib import Path
import torch
import pickle

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

def TopMassComparisonPlots(stacks):
    for r in stacks:
        hist = TH1F()
        hist.Histograms = [k for k in stacks[r].values() if k is not None]
        hist.Title = "Kinematic Region " + r.replace("pt", "pt").replace("eta", "\eta").replace("|", ", ")
        hist.xTitle = "Invariant Top Mass (GeV)"
        hist.yTitle = "Entries / 5 GeV"
        hist.xMin = 50
        hist.xMax = 450
        hist.xBins = 80
        hist.xStep = 40
        hist.OutputDirectory = figure_path + "/topefficiency/top-reconstruction"
        kins = r.replace("|", "<").split("<")
        kins = [f.replace(" ", "") for f in kins]
        hist.Filename = "pt_" + kins[0] + "_" + kins[2] + "_eta_" + kins[3] + "_" + kins[5]
        hist.SaveFigure()

def TopMassComparison(ana, stacks):
    data_p = ana.predicted_topmass
    for r in data_p:
        if r not in stacks: stacks[r] = {"truth" : None, "pred" : None}
        if stacks[r]["pred"] is None:
            stacks[r]["pred"] = TH1F()
            stacks[r]["pred"].Title = "Reconstructed Top Mass"
        stacks[r]["pred"].xData += data_p[r]

    data_t = ana.truth_topmass
    for r in data_t:
        if r not in stacks: stacks[r] = {"truth" : None, "pred" : None}
        if stacks[r]["truth"] is None:
            stacks[r]["truth"] = TH1F()
            stacks[r]["truth"].Title = "Truth Top Mass"
        stacks[r]["truth"].xData += data_t[r]
    return stacks

def TopEfficiency(ana):
    if not isinstance(ana, str):
        ChildrenRegions(ana)
        TruthJetsRegions(ana)
        JetsRegions(ana)


    p = Path(ana)
    files = [str(x) for x in p.glob("**/*.pkl") if str(x).endswith(".pkl")]
    files = list(set(files))
    stacks = {}
    for i in range(len(files)):
        pr = pickle.load(open(files[i], "rb"))
        print(files[i], (i+1) / len(files))
        stacks = TopMassComparison(pr, stacks)
    TopMassComparisonPlots(stacks)

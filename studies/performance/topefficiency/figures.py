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

def TopMassComparison(stacks, ana = None):
    if ana is not None:
        data_p = ana.predicted_topmass
        for r in data_p:
            if r not in stacks: stacks[r] = {"truth" : [], "pred" : []}
            stacks[r]["pred"] += data_p[r]

        data_t = ana.truth_topmass
        for r in data_t:
            if r not in stacks: stacks[r] = {"truth" : [], "pred" : []}
            stacks[r]["truth"] += data_t[r]
        return stacks

    for r in stacks:
        hist_t = TH1F()
        hist_t.Title = "Truth Tops"
        hist_t.xData = stacks[r]["truth"]

        hist_p = TH1F()
        hist_p.Title = "Reconstructed Tops"
        hist_p.xData = stacks[r]["pred"]

        hist = TH1F()
        hist.Histogram = hist_t
        hist.Histograms = [hist_p]
        hist.Title = "Kinematic Region " + r.replace("pt", "pt").replace("eta", "\eta").replace("|", ", ")
        hist.xTitle = "Invariant Top Mass (GeV)"
        hist.yTitle = "Entries / 10 GeV"
        hist.ErrorBars = True
        hist.Style = "ATLAS"
        hist.xMin = 0
        hist.xMax = 1000
        hist.xBins = 100
        hist.xStep = 100
        hist.OutputDirectory = figure_path + "/topefficiency/top-reconstruction"
        kins = r.replace("|", "<").split("<")
        kins = [f.replace(" ", "") for f in kins]
        hist.Filename = "pt_" + kins[0] + "_" + kins[2] + "_eta_" + kins[3] + "_" + kins[5]
        hist.SaveFigure()

def TopMatrix(stacks, ana = None):
    if ana is not None:
        for r in ana.n_tops_predictions:
            if r not in stacks: stacks[r] = {"truth" : [], "pred" : []}
            stacks[r]["pred"] += ana.n_tops_predictions[r]

        for r in ana.n_tops_real:
            if r not in stacks: stacks[r] = {"truth" : [], "pred" : []}
            stacks[r]["truth"] += ana.n_tops_real[r]
        return stacks

    for r in stacks:
        hist = TH2F()
        hist.Title = "Kinematic Region " + r.replace("pt", "pt").replace("eta", "\eta").replace("|", ", ")
        hist.xTitle = "n-Truth Tops"
        hist.yTitle = "n-Reconstructed Tops"
        hist.Color = "tab20c"
        hist.xMin = 0
        hist.xMax = 4
        hist.xBins = 4
        hist.xStep = 1

        hist.yMin = 0
        hist.yMax = 4
        hist.yBins = 4
        hist.yStep = 1

        hist.xData = stacks[r]["truth"]
        hist.yData = stacks[r]["pred"]

        hist.OutputDirectory = figure_path + "/topefficiency/top-matrix"
        kins = r.replace("|", "<").split("<")
        kins = [f.replace(" ", "") for f in kins]
        hist.Filename = "pt_" + kins[0] + "_" + kins[2] + "_eta_" + kins[3] + "_" + kins[5]
        hist.SaveFigure()


def TopEfficiency(ana):
    if not isinstance(ana, str):
        ChildrenRegions(ana)
        TruthJetsRegions(ana)
        JetsRegions(ana)
        return


    p = Path(ana)
    files = [str(x) for x in p.glob("**/*.pkl") if str(x).endswith(".pkl")]
    files = list(set(files))
    stacks = {}
    stacks_t = {}
    for i in range(len(files)):
        pr = pickle.load(open(files[i], "rb"))
        print(files[i], (i+1) / len(files))
        stacks = TopMassComparison(stacks, pr)
        stacks_t = TopMatrix(stacks_t, pr)
    TopMassComparison(stacks)
    TopMatrix(stacks_t)

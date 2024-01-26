from AnalysisG.Plotting import TH1F, TH2F
from dataset_mapping import DataSets
import os

def settings(output):
    setting = {
            "Style" : "ATLAS",
            "ATLASLumi" : None,
            "NEvents" : None,
            "OutputDirectory" : "./Plotting/Dilepton/" + output,
            "Histograms" : [],
            "Histogram" : None,
            "LegendLoc" : "upper right"
    }
    return {i : x for i, x in setting.items()}

def topleptonic(inpt, data):
    smpls = {}
    for dx in inpt.values():
        smpl = data.CheckThis(dx.ROOT)
        if smpl not in smpls: smpls[smpl] = dx
        else: smpls[smpl] += dx

    modes = {
            "Lep-Had" : "Leptonic and Hadronic",
            "Had-Had" : "Hadronic and Hadronic",
            "Lep-Lep" : "Leptonic and Leptonic"
    }

    for mode, title in modes.items():
        hists = []
        for key in smpls:
            hists.append(TH1F())
            hists[-1].Title = key
            hists[-1].xData = smpls[key].TopMasses[mode]

        masses = TH1F()
        masses.Alpha = 0.75
        masses.Style = "ATLAS"
        masses.LegendSize = 1
        masses.LineWidth = 1
        masses.xMin = 0
        masses.xMax = 500
        masses.xBins = 100
        masses.xStep = 100
        masses.Stack = True
        masses.Normalize = True
        masses.OverFlow = True
        masses.Histograms = hists
        masses.Title = "Reconstructed Top Quark Pairs from " + title
        masses.xTitle = "Invariant Top Quark Mass (GeV)"
        masses.yTitle = "Entries"
        masses.Filename = "top-mass-" + mode
        masses.OutputDirectory = "./Output/Dilepton/"
        masses.SaveFigure()

def zprimeleptonic(inpt, data):
    smpls = {}
    for dx in inpt.values():
        smpl = data.CheckThis(dx.ROOT)
        if smpl not in smpls: smpls[smpl] = dx
        else: smpls[smpl] += dx

    modes = {
            "Lep-Had" : "Leptonic and Hadronic",
            "Had-Had" : "Hadronic and Hadronic",
            "Lep-Lep" : "Leptonic and Leptonic"
    }

    for mode, title in modes.items():
        hists = []
        for key in smpls:
            hists.append(TH1F())
            hists[-1].Title = key
            try: hists[-1].xData = smpls[key].ZPrime[""][mode]
            except: hists[-1].xData = []

        masses = TH1F()
        masses.Alpha = 0.75
        masses.Style = "ATLAS"
        masses.LegendSize = 1
        masses.LineWidth = 1
        masses.xMin = 0
        masses.xMax = 2000
        masses.xStep = 200
        masses.xBins = 100
        masses.Stack = True
        masses.Normalize = True
        masses.OverFlow = True
        masses.Histograms = hists
        masses.Title = "Reconstructed Z-Prime Invariant mass derived from \n Top Pairs decaying via " + title
        masses.xTitle = "Invariant Z-Prime Mass (GeV)"
        masses.yTitle = "Entries <unit>"
        masses.Filename = "zprime-mass-" + mode
        masses.OutputDirectory = "./Output/Dilepton/"
        masses.SaveFigure()

def doubleleptonic_Plotting(inpt, path):
    data = DataSets(path)
    sm = None
    for i in inpt:
        if sm is None: sm = inpt[i]
        else: sm += inpt[i]

    passed = sm.CutFlow["Selection::Passed"]
    rejected = sm.CutFlow["Selection::Rejected"]
    solsFound = sm.CutFlow["Strategy::FoundSolutions::Passed"]
    nosols = sm.CutFlow["Strategy::NoDoubleNuSolution::Rejected"]
    all_ = passed + rejected
    print("Strategy Passed:", 100*passed/all_, "%")
    print("Strategy Rejected:", 100*rejected/all_, "%")
    print("Strategy Passed (Solutions found):", 100*solsFound/all_, "%")
    print("Strategy Passed (No Solutions found):", 100*nosols/all_, "%")

    #topleptonic(inpt, data)
    zprimeleptonic(inpt, data)
    #print(inpt.PhaseSpaceZ)








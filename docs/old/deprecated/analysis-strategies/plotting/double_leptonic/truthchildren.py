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

def topleptonic(smpls):
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

def zprimeleptonic(smpls):
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

def topleptonic_phasespace(smpls, kinematic):
    njets_sign = {}
    for key in smpls:
        if kinematic == "mass": container = smpls[key].PhaseSpaceT[""]
        else: container = smpls[key].Kinematics[""]
        for nj in container:
            for sign in container[nj]:
                mode = ""
                if sign == "++" or sign == "--": mode = "Same Sign"
                elif sign == "-" or sign == "+": mode = "Single Lepton"
                elif sign == "+-" or sign == "-+": mode = "Opposite Sign"
                elif sign == "NA": mode = "Hadronic Pair"
                else: sign = "WARNING!"

                if mode not in njets_sign: njets_sign[mode] = {}
                if nj not in njets_sign[mode]: njets_sign[mode][nj] = {}
                if key not in njets_sign[mode][nj]: njets_sign[mode][nj][key] = []
                if kinematic == "mass": njets_sign[mode][nj][key] += container[nj][sign]
                else: njets_sign[mode][nj][key] += container[nj][sign][kinematic]

    hists = {}
    for sign in njets_sign:
        for nj in njets_sign[sign]:
            if sign not in hists: hists[sign] = {}
            if nj not in hists[sign]: hists[sign][nj] = []
            for prc in njets_sign[sign][nj]:
                h = TH1F()
                h.xData = njets_sign[sign][nj][prc]
                h.Title = prc
                hists[sign][nj] += [h]

    for sign in hists:
        for nj in hists[sign]:
            plt = TH1F()
            plt.Alpha = 0.75
            plt.LegendSize = 1
            plt.Style = "ATLAS"
            plt.xBins = 100
            plt.Stack = True
            plt.OverFlow = True
            plt.Histograms = hists[sign][nj]
            if kinematic == "mass":
                plt.Title = "Reconstructed Invariant Top (pairs) Masses in \n " + sign + " with n-Jets: " + str(nj)
                plt.xTitle = "Invariant Top-Mass (GeV)"
                plt.Filename = "top-masses-njets-" + str(nj)
                plt.xMin = 0
                plt.xMax = 1000
                plt.xStep = 100
            else:
                plt.Title = "Particle Kinematics (" + kinematic.upper() + ") in \n " + sign + " with n-Jets: " + str(nj)
                plt.xTitle = kinematic
                plt.Filename = "particle-" + kinematic + "-njets-" + str(nj)

            if kinematic == "pT" or kinematic == "Energy":
                plt.xMin = 0
                plt.xMax = 1000
                plt.xStep = 100
                plt.xTitle += " (GeV)"

            if kinematic == "Eta":
                plt.xMin = -3
                plt.xMax = 3
                plt.xStep = 1

            if kinematic == "b-Jets":
                plt.xMin = 0
                plt.xMax = 10
                plt.xStep = 10
                plt.xTitle = "b-Jets"

            plt.yTitle = "Entries"
            plt.OutputDirectory = "./Output/Dilepton/" + kinematic + "/regions/" + sign.replace(" ", "-").lower()
            plt.SaveFigure()


def doubleleptonic_Plotting(sm, smpls):
    passed = sm.CutFlow["Selection::Passed"]
    rejected = sm.CutFlow["Selection::Rejected"]
    solsFound = sm.CutFlow["Strategy::FoundSolutions::Passed"]
    nosols = sm.CutFlow["Strategy::NoDoubleNuSolution::Rejected"]
    all_ = passed + rejected
    print("Strategy Passed:", 100*passed/all_, "%")
    print("Strategy Rejected:", 100*rejected/all_, "%")
    print("Strategy Passed (Solutions found):", 100*solsFound/all_, "%")
    print("Strategy Passed (No Solutions found):", 100*nosols/all_, "%")

    #topleptonic(smpls)
    #zprimeleptonic(smpls)
    #topleptonic_phasespace(smpls, "mass")
    #topleptonic_phasespace(smpls, "pT")
    #topleptonic_phasespace(smpls, "Eta")
    #topleptonic_phasespace(smpls, "Energy")
    #topleptonic_phasespace(smpls, "b-Jets")

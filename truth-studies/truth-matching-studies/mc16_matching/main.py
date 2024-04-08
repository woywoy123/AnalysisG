from AnalysisG.Events import Event
from AnalysisG.Tools import Tools
from AnalysisG import Analysis
import plotting
import studies
import os

figure_path = "../../../docs/source/studies/truth-matching/mc16/"
plotting.resonance.zprime.figure_path            = figure_path
plotting.resonance.decaymodes.figure_path        = figure_path
plotting.top.topkinematics.figure_path           = figure_path
plotting.top.topmatching.figure_path             = figure_path
plotting.children.childrenkinematics.figure_path = figure_path
plotting.truthjets.toptruthjets.figure_path      = figure_path
plotting.jets.topmatching.figure_path            = figure_path
plotting.event.figure_path                       = figure_path

smpls = os.environ["Samples"] + "mc16_13_"
run_cache = False
run_analysis = {
                    "ZPrimeMatrix"        : studies.resonance.zprime.ZPrime,
                    "ResonanceDecayModes" : studies.resonance.decaymodes.DecayModes,
                    "TopKinematics"       : studies.top.topkinematics.TopKinematics,
                    "TopMatching"         : studies.top.topmatching.TopMatching,
                    "ChildrenKinematics"  : studies.children.childrenkinematics.ChildrenKinematics,
                    "TopTruthJets"        : studies.truthjets.toptruthjets.TopTruthJets,
                    "TopJets"             : studies.jets.topmatching.TopJets,
                    "TruthEvent"          : studies.event.TruthEvent,
                    "AddOnStudies"        : studies.other.AddOnStudies
}

run_plotting = {
                    "ZPrimeMatrix"        : plotting.resonance.zprime.ZPrime,
                    "ResonanceDecayModes" : plotting.resonance.decaymodes.DecayModes,
                    "TopKinematics"       : plotting.top.topkinematics.TopKinematics,
                    "TopMatching"         : plotting.top.topmatching.TopMatching,
                    "ChildrenKinematics"  : plotting.children.childrenkinematics.ChildrenKinematics,
                    "TopTruthJets"        : plotting.truthjets.toptruthjets.TopTruthJets,
                    "TopJets"             : plotting.jets.topmatching.TopJets,
                    "TruthEvent"          : plotting.event.TruthEvent,
                    "AddOnStudies"        : plotting.other.AddOnStudies
}



smpls = {
        "m400"  : smpls + "ttZ_m400",
        "m500"  : smpls + "ttZ_m500",
        "m600"  : smpls + "ttZ_m600",
        "m700"  : smpls + "ttZ_m700",
        "m800"  : smpls + "ttZ_m800",
        "m900"  : smpls + "ttZ_m900",
        "m1000" : smpls + "ttZ_m1000"
}

x = Tools()
for bsm in smpls:
    bs = {bsm : x.lsFiles(smpls[bsm])}
    if run_cache:
        ana = Analysis()
        ana.ProjectName = bsm
        for j, k in bs.items(): ana.InputSample(j, k)
        ana.EventCache = True
        ana.Event = Event
        ana.Threads = 24
        ana.Chunks = 1000
        ana.EventStop = 20000
        ana.Launch()

    if len(run_analysis):
        ana = Analysis()
        ana.ProjectName = bsm
        ana.EventCache = True
        ana.EventName = "Event"
        for i, j in run_analysis.items():
            ana.AddSelection(j)
            ana.This(j.__name__, "nominal")
        ana.Threads = 24
        ana.Chunks = 1000
        ana.Launch()

    for i, j in run_plotting.items():
        ana = Analysis()
        ana.ProjectName = bsm
        j(ana.merged["nominal." + j.__name__])

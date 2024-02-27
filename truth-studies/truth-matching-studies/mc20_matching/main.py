from AnalysisG.Events import SSML_MC20
from AnalysisG import Analysis
import plotting
import studies
import os

smpls = os.environ["Samples"] + "mc20_13_tttt_m1250"
run_cache = True
run_analysis = {
        "truth-top" : studies.truthtop.TruthTops,
        "truth-jets" : studies.truthjet.TruthJetMatching
}

run_plotting = {
        "truth-top" : plotting.truthtop.TruthTops,
        "truth-jets" : plotting.truthjet.TruthJets
}

if run_cache:
    ana = Analysis()
    ana.InputSample(None, smpls)
    ana.Verbose = 3
    ana.EventCache = True
    ana.Event = SSML_MC20
    ana.Launch()

if "truth-top" in run_analysis:
    ana = Analysis()
    ana.InputSample(None)
    ana.EventCache = True
    ana.EventName = "SSML_MC20"
    ana.AddSelection(run_analysis["truth-top"])
    ana.This("TruthTops", "nominal_Loose")
    ana.Launch()

if "truth-jets" in run_analysis:
    ana = Analysis()
    ana.EventCache = True
    ana.EventName = "SSML_MC20"
    ana.AddSelection(run_analysis["truth-jets"])
    ana.This("TruthJetMatching", "nominal_Loose")
    ana.Launch()

if "truth-top" in run_plotting:
    ana = Analysis()
    run_plotting["truth-top"](ana.merged["nominal_Loose.TruthTops"])

if "truth-jets" in run_plotting:
    ana = Analysis()
    run_plotting["truth-jets"](ana.merged["nominal_Loose.TruthJetMatching"])

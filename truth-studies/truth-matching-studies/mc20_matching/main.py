from AnalysisG.Events import SSML_MC20
from AnalysisG import Analysis
import plotting
import studies
import os

smpls = os.environ["Samples"] + "mc20_13_tttt_m1250"
run_cache = False
run_analysis = {
#        "truth-top" : studies.truthtop.TruthTops
}

run_plotting = {
        "truth-top" : plotting.truthtop.TruthTops
}

if run_cache:
    ana = Analysis()
    ana.InputSample(None, smpls)
    ana.Verbose = 3
    ana.DebugMode = True
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
    ana.Threads = 1
    ana.Launch()

if "truth-top" in run_plotting:
    ana = Analysis()
    run_plotting["truth-top"](ana.merged["nominal_Loose.TruthTops"])

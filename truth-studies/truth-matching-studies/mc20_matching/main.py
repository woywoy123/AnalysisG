from AnalysisG.Events import SSML_MC20
from AnalysisG.Tools import Tools
from AnalysisG import Analysis
import plotting
import studies
import os

smpls = os.environ["Samples"] + "mc20_13_tttt_m1250"
run_cache = False
run_analysis = {
        #"TruthTops" : studies.truthtop.TruthTops,
        #"TruthJetMatching" : studies.truthjet.TruthJetMatching
}

run_plotting = {
        "TruthTops" : plotting.truthtop.TruthTops,
        #"TruthJetMatching" : plotting.truthjet.TruthJetMatching
}

x = Tools()
bs = {"test" : x.lsFiles(smpls)}
bsm = "UNTITLED"
if run_cache:
    ana = Analysis()
    ana.Chunks = 1000
    ana.EventStop = 12000
    ana.ProjectName = bsm
    for j, k in bs.items(): ana.InputSample(j, k)
    ana.EventCache = True
    ana.Event = SSML_MC20
    ana.Launch()

for i, j in run_analysis.items():
    ana = Analysis()
    ana.Chunks = 1000
    ana.ProjectName = bsm
    ana.AddSelection(j)
    ana.EventCache = True
    ana.EventName = "SSML_MC20"
    ana.This(j.__name__, "nominal_Loose")
    ana.Launch()

for i, j in run_plotting.items():
    ana = Analysis()
    ana.ProjectName = bsm
    j(ana.merged["nominal_Loose." + j.__name__])

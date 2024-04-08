from AnalysisG.Events import SSML_MC20
from AnalysisG.Tools import Tools
from AnalysisG import Analysis
import plotting
import studies
import os

figure_path = "../../../docs/source/studies/truth-matching/mc20/"
plotting.top.figure_path = figure_path

smpls = os.environ["Samples"] + "mc20"
run_cache = True
run_analysis = {
        "TopMatching" : studies.top.TopMatching,
}

run_plotting = {
        "TopMatching" : plotting.top.TopMatching,
}

x = Tools()
bs = {"test" : x.lsFiles(smpls)[:10]}
bsm = "UNTITLED"
if run_cache:
    ana = Analysis()
    ana.Chunks = 1000
    ana.EventStop = 10000
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

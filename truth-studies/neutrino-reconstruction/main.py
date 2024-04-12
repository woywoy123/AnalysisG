from AnalysisG import Analysis
from AnalysisG.Events import Event
import studies
import plots
import os

figure_path = "../../docs/source/studies/"
plots.nu_valid.figure_path = figure_path

run_analysis = {
    "single-nu" : studies.nu_valid.NeutrinoReconstruction,
}

run_plotting = {
    "single-nu" : plots.nu_valid.NeutrinoReconstruction,
}

build_cache = False
if build_cache:
    smpls = os.environ["Samples"] + "single_lep/"
    ana = Analysis()
    ana.ProjectName = "neutrino"
    ana.InputSample(None, smpls)
    ana.EventCache = True
    ana.Event = Event
    ana.Chunks = 1000
    ana.EventStop = 10000
    ana.Launch()

if len(run_analysis):
    ana = Analysis()
    ana.ProjectName = "neutrino"
    ana.EventName = "Event"
    for i, j in run_analysis.items():
        ana.AddSelection(j)
        ana.This(j.__name__, "nominal")
    ana.EventCache = True
    ana.Chunks = 1000
    ana.Threads = 1
    ana.Launch()

for i, j in run_plotting.items():
    ana = Analysis()
    ana.ProjectName = "neutrino"
    j(ana.merged["nominal." + j.__name__])

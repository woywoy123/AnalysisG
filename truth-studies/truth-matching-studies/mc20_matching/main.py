from AnalysisG.Events import SSML_MC20
from AnalysisG.Tools import Tools
from AnalysisG import Analysis
import plotting
import studies
import os

figure_path = "../../../docs/source/studies/truth-matching/mc20/"
plotting.top.figure_path = figure_path
plotting.children.figure_path = figure_path
smpls = os.environ["Samples"] + "Binbin"
run_cache = False
run_analysis = {
#        "TopMatching" : studies.top.TopMatching,
#        "Children-Kinematics" : studies.children.ChildrenKinematics,
}

run_plotting = {
#        "TopMatching" : plotting.top.TopMatching,
        "Children-Kinematics" : plotting.children.ChildrenKinematics,
}

smple_dr = {
        "dr-0.05" : [], "dr-0.10" : [], "dr-0.15" : [],
        "dr-0.20" : [], "dr-0.30" : [], "dr-0.40" : [],
        "broken-jets" : []
}
x = Tools()
for i in x.lsFiles(smpls):
    if "_dR0p05_" in i: smple_dr["dr-0.05"] += [i]
    elif "_dR0p1_" in i: smple_dr["dr-0.10"] += [i]
    elif "_dR0p15_" in i: smple_dr["dr-0.15"] += [i]
    elif "_dR0p2_" in i: smple_dr["dr-0.20"] += [i]
    elif "_dR0p3_" in i: smple_dr["dr-0.30"] += [i]
    elif "_dR0p4_" in i: smple_dr["dr-0.40"] += [i]
    else: smple_dr["broken-jets"] += [i]

for bsm in smple_dr:
    if run_cache:
        ana = Analysis()
        ana.Chunks = 1000
        ana.EventStop = 10000
        ana.ProjectName = bsm
        ana.InputSample(bsm, smple_dr[bsm][:20])
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
#        plotting.top.figure_path = figure_path + bsm
        j(ana.merged["nominal_Loose." + j.__name__])
    exit()

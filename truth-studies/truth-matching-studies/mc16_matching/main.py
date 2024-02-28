from AnalysisG.Events import Event
from AnalysisG import Analysis
import plotting
import studies
import os
from AnalysisG.Tools import Tools


smpls = os.environ["Samples"]
smpls = {
#        "m400" : smpls + "mc16_13_ttZ_m400",
#        "m500" : smpls + "mc16_13_ttZ_m500",
#        "m600" : smpls + "mc16_13_ttZ_m600",
#        "m700" : smpls + "mc16_13_ttZ_m700",
#        "m800" : smpls + "mc16_13_ttZ_m800",
#        "m900" : smpls + "mc16_13_ttZ_m900",
        "m1000" : smpls + "mc16_13_ttZ_m1000"
}

x = Tools()
smpls = {i : x.lsFiles(smpls[i])[:4] for i in smpls}

run_cache = False
run_analysis = {
#    "ZPrimeMatrix": studies.resonance.zprime.ZPrime,
#    "ResonanceDecayModes" : studies.resonance.decaymodes.DecayModes,
#    "TopKinematics" : studies.top.topkinematics.TopKinematics,
    "TopMatching" : studies.top.topmatching.TopMatching
}

run_plotting = {
#    "ZPrimeMatrix" : plotting.resonance.zprime.ZPrime,
#    "ResonanceDecayModes" : plotting.resonance.decaymodes.DecayModes,
#    "TopKinematics" : plotting.top.topkinematics.TopKinematics
    "TopMatching" : plotting.top.topmatching.TopMatching
}


if run_cache:
    ana = Analysis()
    for j, k in smpls.items(): ana.InputSample(j, k)
    ana.EventCache = True
    ana.Event = Event
    ana.Launch()

for i, j in run_analysis.items():
    ana = Analysis()
    ana.AddSelection(j)
    ana.EventCache = True
    ana.EventName = "Event"
    ana.This(j.__name__, "nominal")
    ana.Launch()

for i, j in run_plotting.items():
    ana = Analysis()
    j(ana.merged["nominal." + j.__name__])

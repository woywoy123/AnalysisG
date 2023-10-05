from plotting.double_leptonic.truthchildren import doubleleptonic_Plotting
from AnalysisG.IO import nTupler, PickleObject, UnpickleObject
from double_leptonic import DiLeptonic
from AnalysisG.Events import Event
from AnalysisG import Analysis
import os


smpls = os.environ["Samples"]
which_analysis = "double-leptonic:DiLeptonic"
_ana = which_analysis.split(":")
if _ana[0] == "double-leptonic": ana = DiLeptonic()

combinations = [
        ("is_b", "children"),
        ("btag_DL1r_60", "jets+truthleptons"),
        ("btag_DL1r_77", "jets+truthleptons"),
        ("btag_DL1r_85", "jets+truthleptons"),

        ("btag_DL1_60", "jets+truthleptons"),
        ("btag_DL1_77", "jets+truthleptons"),
        ("btag_DL1_85", "jets+truthleptons"),

        ("btag_DL1r_60", "detector"),
        ("btag_DL1r_77", "detector"),
        ("btag_DL1r_85", "detector"),

        ("btag_DL1_60",  "detector"),
        ("btag_DL1_77",  "detector"),
        ("btag_DL1_85",  "detector")
]

samples = {
        "ttZ"   : smpls + "ttZ-1000",
        "other" : smpls + "other",
        "t"     : smpls + "t"    ,
        "tt"    : smpls + "tt"   ,
        "ttbar" : smpls + "ttbar",
        "ttH"   : smpls + "ttH"  ,
        "tttt"  : smpls + "tttt" ,
        "ttX"   : smpls + "ttX"  ,
        "ttXll" : smpls + "ttXll",
        "ttXqq" : smpls + "ttXqq",
        "V"     : smpls + "V"    ,
        "Vll"   : smpls + "Vll"  ,
        "Vqq"   : smpls + "Vqq"
}


for pairs in combinations:
    tagger, truth = pairs
    ana.__params__ = { "btagger" : tagger , "truth" : truth}
    run = True
    pkl = True
    name = _ana[1] + "_" + tagger + "_" + truth.replace("+","-")
    if run:
        Ana = Analysis()
        Ana.ProjectName = _ana[1]
        for key in samples: Ana.InputSample(key, samples[key])
        Ana.AddSelection(ana)
        Ana.EventCache = True
        Ana.Event = Event
        Ana.Threads = 12
        Ana.Chunks = 1000
        Ana.Launch()

    if pkl:
        tupler = nTupler()
        tupler.Threads = 12
        tupler.Chunks = 1000
        tupler.ProjectName = _ana[1]
        tupler.This("DiLeptonic -> ", "nominal")

        x = tupler.merged()
        tupler.rm(tupler.WorkingPath + "/SelectionCache")
        PickleObject(x["nominal.DiLeptonic"].__getstate__(), name)

    ana.__setstate__(UnpickleObject(name))
    print("Building Plots: " + name)
    doubleleptonic_Plotting(ana, name)





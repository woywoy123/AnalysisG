from plotting.double_leptonic.truthchildren import doubleleptonic_Plotting
from AnalysisG.IO import nTupler, PickleObject, UnpickleObject
from AnalysisG.Events import Event, SSML
from double_leptonic import DiLeptonic
from dataset_mapping import DataSets
from AnalysisG.Tools import Tools
from AnalysisG import Analysis
import os

smpls = os.environ["Samples"]
which_analysis = "double-leptonic:DiLeptonic"
_ana = which_analysis.split(":")
if _ana[0] == "double-leptonic": ana = DiLeptonic()

combinations = [
        ("gn2_btag_85", "detector"),
]

samples = {
    "SSML" : smpls
}


t = Tools()
all_smpls = []
for key in samples: all_smpls += t.lsFiles(samples[key])
quant = list(t.Quantize(all_smpls, 2))
run_cache = False
run_selection = False
run_ntpl = False
tree = "nominal_Loose"

x = 0
for smpl in quant:
    if run_cache: pass
    else: break
    Ana = Analysis()
    Ana.ProjectName = _ana[1]
    Ana.InputSample("smpl-" + str(x), smpl)
    Ana.EventCache = True
    Ana.SelectionName = None
    Ana.Event = SSML()
    Ana.Threads = 12
    Ana.Chunks = 10000
#    Ana.EventStop = 20000
    Ana.Launch()
    x+=1

for pairs in combinations:
    tagger, truth = pairs
    ana.__params__ = { "btagger" : tagger , "truth" : truth}
    name = _ana[1] + "_" + tagger + "_" + truth.replace("+","-")
    x = 0
    for smpl in quant:
        if run_selection: pass
        else: break
        Ana = Analysis()
        Ana.ProjectName = _ana[1]
        Ana.InputSample("smpl-" + str(x))
        Ana.AddSelection(ana)
        Ana.EventName = "SSML"
        Ana.EventCache = True
        Ana.Threads = 12
        Ana.MaxRam = 16
        Ana.Chunks = 1000
        Ana.EventStop = 10000
        Ana.Launch()
        x += 1

    x = 0
    for smpl in quant:
        if run_ntpl: pass
        else: break
        Ana = Analysis()
        Ana.ProjectName = _ana[1]
        Ana.InputSample("smpl-" + str(x))
        Ana.This("DiLeptonic", tree)
        Ana.Threads = 12
        Ana.Chunks = 10000
        Ana.Launch()
        x += 1

    #print("Reading pickle")
    #print("Building Plots: " + name)
    #Ana = Analysis()
    #Ana.ProjectName = _ana[1]
    #Ana.AddSelection(ana)
    #output = Ana.merged

    #sm = None
    #lex, ix = len(output), 0
    #data = DataSets(samples["SSML"])
    #for i in output:
    #    if sm is None: sm = output[i]
    #    else: sm += output[i]
    #    print(i, float(ix/lex)*100)
    #    ix += 1
    #PickleObject(sm.__getstate__(), "sm")

    #print("------")
    #ix = 0
    #smpls = {}
    #for dx in output:
    #    smpl = data.CheckThis(output[dx].ROOT)
    #    if smpl not in smpls: smpls[smpl] = output[dx]
    #    else: smpls[smpl] += output[dx]
    #    print(smpl, float(ix/lex)*100)
    #    ix += 1

    #smpls = {k : smpls[k].__getstate__() for k in smpls}
    #PickleObject(smpls, "data")

    sm = UnpickleObject("sm")
    o = DiLeptonic()
    o.__setstate__(sm)

    smplsx = {}
    smpls = UnpickleObject("data")
    for k in smpls:
        smplsx[k] = DiLeptonic()
        smplsx[k].__setstate__(smpls[k])
    doubleleptonic_Plotting(o, smplsx)

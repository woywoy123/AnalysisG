from AnalysisG.generators import Analysis
from AnalysisG.core import IO
import pickle
import pathlib

def chunks(lst, n):
    for i in range(0, len(lst), n): yield lst[i:i + n]

def build_samples(pth, pattern, chnks):
    return chunks([str(i) for i in pathlib.Path(pth).glob(pattern) if str(i).endswith(".root")], chnks)

# model implementations
from AnalysisG.models import *
model_method = {"RecursiveGraphNeuralNetwork" : RecursiveGraphNeuralNetwork, "Grift" : Grift}

# selection studies
from AnalysisG.selections.performance.topefficiency.topefficiency import TopEfficiency
selection_method = {"TopEfficiency" : TopEfficiency}
selection_container = {}

# event implementations
from AnalysisG.events.gnn import EventGNN
from AnalysisG.events.bsm_4tops import BSM4Tops
event_method = {"BSM4Tops" : BSM4Tops, "EventGNN" : EventGNN}

# graph implementations
from AnalysisG.graphs.bsm_4tops import *
graph_method = {
        "GraphTruthJets"     : GraphTruthJets,
        "GraphTruthJetsNoNu" : GraphTruthJetsNoNu,
        "GraphJets"          : GraphJets,
        "GraphJetsNoNu"      : GraphJetsNoNu,
        "GraphDetectorLep"   : GraphDetectorLep,
        "GraphDetector"      : GraphDetector
}

# figures
import topefficiency
plotting_method = {"topefficiency" : topefficiency}

figure_path = "./Output/"
study       = "topefficiency"

graph_name     = "GraphJets"
graph_prefix   = "_bn_1"
inference_mode = False
plot_only      = False
fetch_meta     = False
threads        = 8

graph_cache = "/scratch/tnom6927/graph-data-mc16-full/"
root_model  = "/import/wu1/tnom6927/TrainingOutput/training-sessions/"
#data_path   = "/import/wu1/tnom6927/TrainingOutput/mc16-full/" # <----- cluster
#data_path = "/CERN/Samples/mc16-full" # <----- local
data_path = "/CERN/trainings/mc16-full-inference/ROOT/" + graph_name + graph_prefix + "_Grift/MRK-1/epoch-1/kfold-1/"

model_states = {}
if inference_mode:
    model_states |= {
            "MRK-1" : {"epoch-" + str(ep) : (["kfold-1"], ["cuda:1"]) for ep in [1, 2, 3, 4]},
            "MRK-2" : {"epoch-" + str(ep) : (["kfold-1"], ["cuda:0"]) for ep in [1, 2, 3, 4]},
            "MRK-3" : {"epoch-" + str(ep) : (["kfold-1"], ["cuda:1"]) for ep in [1, 2, 3, 4]},
            "MRK-4" : {"epoch-" + str(ep) : (["kfold-1"], ["cuda:0"]) for ep in [1, 2, 3, 4]},
            "MRK-5" : {"epoch-" + str(ep) : (["kfold-1"], ["cuda:1"]) for ep in [1, 2, 3, 4]},
            "MRK-6" : {"epoch-" + str(ep) : (["kfold-1"], ["cuda:0"]) for ep in [1, 2, 3, 4]},
            "MRK-7" : {"epoch-" + str(ep) : (["kfold-1"], ["cuda:1"]) for ep in [1, 2, 3, 4]},
            "MRK-8" : {"epoch-" + str(ep) : (["kfold-1"], ["cuda:0"]) for ep in [1, 2, 3, 4]},
    }

sample_path_ = {
        "singletop"      : data_path + "/singletop"  + "/*",
        "ttH125"         : data_path + "/ttH125"     + "/*",
        "ttbarHT1k"      : data_path + "/ttbarHT1k"  + "/*",
        "SM4topsNLO"     : data_path + "/SM4topsNLO" + "/*",
        "ttbar"          : data_path + "/ttbar"      + "/*",
        "ttbarHT1k5"     : data_path + "/ttbarHT1k5" + "/*",
        "ttbarHT6c"      : data_path + "/ttbarHT6c"  + "/*",
        "Ztautau"        : data_path + "/Ztautau"    + "/*",
        "llll"           : data_path + "/llll"       + "/*",
        "lllv"           : data_path + "/lllv"       + "/*",
        "llvv"           : data_path + "/llvv"       + "/*",
        "lvvv"           : data_path + "/lvvv"       + "/*",
        "tchan"          : data_path + "/tchan"      + "/*",
        "tt"             : data_path + "/tt"         + "/*",
        "ttee"           : data_path + "/ttee"       + "/*",
        "ttmumu"         : data_path + "/ttmumu"     + "/*",
        "tttautau"       : data_path + "/tttautau"   + "/*",
        "ttW"            : data_path + "/ttW"        + "/*",
        "ttZnunu"        : data_path + "/ttZnunu"    + "/*",
        "ttZqq"          : data_path + "/ttZqq"      + "/*",
        "tW"             : data_path + "/tW"         + "/*",
        "tZ"             : data_path + "/tZ"         + "/*",
        "Wenu"           : data_path + "/Wenu"       + "/*",
        "WH125"          : data_path + "/WH125"      + "/*",
        "WlvZqq"         : data_path + "/WlvZqq"     + "/*",
        "Wmunu"          : data_path + "/Wmunu"      + "/*",
        "WplvWmqq"       : data_path + "/WplvWmqq"   + "/*",
        "WpqqWmlv"       : data_path + "/WpqqWmlv"   + "/*",
        "WqqZll"         : data_path + "/WqqZll"     + "/*",
        "WqqZvv"         : data_path + "/WqqZvv"     + "/*",
        "Wt"             : data_path + "/Wt"         + "/*",
        "Wtaunu"         : data_path + "/Wtaunu"     + "/*",
        "Zee"            : data_path + "/Zee"        + "/*",
        "ZH125"          : data_path + "/ZH125"      + "/*",
        "Zmumu"          : data_path + "/Zmumu"      + "/*",
        "ZqqZll"         : data_path + "/ZqqZll"     + "/*",
        "ZqqZvv"         : data_path + "/ZqqZvv"     + "/*",
        "ttH_tttt_m400"  : data_path + "/tttt_m400"  + "/*",
        "ttH_tttt_m500"  : data_path + "/tttt_m500"  + "/*",
        "ttH_tttt_m600"  : data_path + "/tttt_m600"  + "/*",
        "ttH_tttt_m700"  : data_path + "/tttt_m700"  + "/*",
        "ttH_tttt_m800"  : data_path + "/tttt_m800"  + "/*",
        "ttH_tttt_m900"  : data_path + "/tttt_m900"  + "/*",
        "ttH_tttt_m1000" : data_path + "/tttt_m1000" + "/*",
}

for i in sample_path_:
    if not fetch_meta: break
    ana = Analysis()
    ana.AddSamples(sample_path_[i], i)
    ana.FetchMeta = True
    ana.SumOfWeightsTreeName = "sumWeights"
    ana.Start()
if fetch_meta: exit()

if not inference_mode:
    ls = sum(list(build_samples(data_path, "./*/*.root", 10)), [])
    sample_path_ = {k.split("/")[-1] : k for k in ls}

event_name     = "BSM4Tops"      if inference_mode     else "EventGNN"
graph_name     = graph_name      if inference_mode     else ""
model_name     = "Grift"         if inference_mode     else ""
selection_name = "TopEfficiency" if not inference_mode else ""

for k in build_samples(data_path, "./*/*/*.root", 10):
    sample_path = {}
    for l in k:
        for t in sample_path_:
            if "/" + t + "/" not in l: continue
            if t not in sample_path: sample_path[t] = []
            sample_path[t] += [l]
            break
    if not len(sample_path): continue

    ana = Analysis()
    ana.Threads = threads
    ana.GraphCache = graph_cache
    if len(model_states): ana.FetchMeta = False
    for j in sample_path:
        for p in sample_path[j]:
            label = j + p.split("/")[-1]
            ana.AddSamples(p, label)

            try: ana.AddEvent(event_method[event_name](), label)
            except KeyError: pass
            except NameError: pass

            try: ana.AddGraph(graph_method[graph_name](), label)
            except KeyError: pass
            except NameError: pass

    for j in model_states:
        for ep in model_states[j]:
            mdl = root_model + graph_name + graph_prefix + "/" + model_name + "/" + j + "/state/" + ep + "/"
            kf, dev = model_states[j][ep]
            for i in range(len(kf)):
                gnn = model_method[model_name]()
                gnn.o_edge  = {"top_edge" : "CrossEntropyLoss", "res_edge" : "CrossEntropyLoss"}
                gnn.o_graph = {"ntops"    : "CrossEntropyLoss", "signal"   : "CrossEntropyLoss"}
                gnn.i_node  = ["pt", "eta", "phi", "energy"]
                gnn.device  = dev[i]
                gnn.checkpoint_path = mdl + kf[i] + "_model.pt"
                name = "ROOT/" + graph_name + graph_prefix + "_" + model_name + "/" + j + "/" + ep + "/" + kf[i]
                ana.AddModelInference(gnn, name)
    ana.Start()
    del ana
    sample_path = {}

i = 0
for j, k in sample_path_.items():
    if plot_only: break
    ana = Analysis()
    ana.Threads = 1
    ana.AddSamples(k, j)
    try: ana.AddEvent(event_method[event_name](), j)
    except KeyError: pass
    except NameError: pass

    try: selection_container[selection_name] = selection_method[selection_name]()
    except KeyError: continue
    except NameError: continue
    ana.AddSelection(selection_container[selection_name])
    ana.Start()

    f = open("./serialized-data/" + j +".pkl", "wb")
    pickle.dump(selection_container[selection_name], f)
    f.close()
    print(i, len(sample_path_))
    i+=1

method = plotting_method[study]
method.figures.figure_path = figure_path
if study == "topefficiency": method.figures.TopEfficiency("./serialized-data/")

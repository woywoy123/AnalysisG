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
graph_prefix   = "_bn_1_"
inference_mode = False
plot_only      = True
fetch_meta     = False
build_cache    = False
threads        = 1
bts            = 100
kfold          = "kfold-5"
epoch          = "epoch-24"
mrk            = "MRK-1"


graph_cache = "/scratch/tnom6927/graph-data-mc16-full/"
root_model  = "/import/wu1/tnom6927/TrainingOutput/training-sessions/"
data_path   = "/import/wu1/tnom6927/TrainingOutput/grid-data/" # <----- cluster
#data_path = "/CERN/trainings/mc16-full-inference" # <----- local
data_path = "ProjectName/ROOT/" + graph_name + graph_prefix + "Grift/" + mrk + "/" + epoch + "/" + kfold + "/"
#data_path = "/CERN/Samples/mc16-full/"

epx = [i+1 for i in range(1, 30)]
model_states = {
    "MRK-1" : {"epoch-" + str(ep) : (["kfold-1", "kfold-5"], ["cuda:1", "cuda:0"]) for ep in epx},
    "MRK-2" : {"epoch-" + str(ep) : (["kfold-1", "kfold-5"], ["cuda:0", "cuda:1"]) for ep in epx},
    "MRK-3" : {"epoch-" + str(ep) : (["kfold-1", "kfold-5"], ["cuda:1", "cuda:0"]) for ep in epx},
    "MRK-4" : {"epoch-" + str(ep) : (["kfold-1", "kfold-5"], ["cuda:0", "cuda:1"]) for ep in epx},
}

ls = list(build_samples(data_path, "**/*.root", 20))
if fetch_meta:
    x = 0
    for i in ls:
        ana = Analysis()
        for k in i: ana.AddSamples(k, k.split("/")[-1])
        ana.FetchMeta = True
        ana.SumOfWeightsTreeName = "sumWeights"
        ana.Start()
        print(x, len(ls))
        x+=1
    exit()

event_name     = "BSM4Tops"      if inference_mode     or build_cache else "EventGNN"
graph_name     = graph_name      if inference_mode     or build_cache else ""
model_name     = "Grift"         if inference_mode     or build_cache else ""
selection_name = "TopEfficiency" if not inference_mode and not build_cache else ""
pth = "./serialized-data/" + mrk + "/" + epoch + "/" + kfold + "/"

i = 0
for k in ls:
    if plot_only: break
    ana = Analysis()
    ana.BuildCache = build_cache
    ana.Threads = threads
    ana.BatchSize = bts
    ana.GraphCache = graph_cache
    if len(model_states): ana.FetchMeta = False

    try:
        selection_container[selection_name] = selection_method[selection_name]()
        try: open(pth + str(i), "r")
        except: ana.AddSelection(selection_container[selection_name])
    except: pass

    for j in k:
        label = j.split("/")[-1]
        ana.AddSamples(j, label)

        try: ana.AddEvent(event_method[event_name](), label)
        except KeyError: pass
        except NameError: pass

        try: ana.AddGraph(graph_method[graph_name](), label)
        except KeyError: pass
        except NameError: pass

    x = 0
    for j in model_states:
        if not inference_mode: break
        for ep in model_states[j]:
            mdl = root_model + graph_name + graph_prefix[:-1] + "/" + model_name + "/" + j + "/state/" + ep + "/"
            kf, dev = model_states[j][ep]
            for k in range(len(kf)):
                gnn = model_method[model_name]()
                gnn.o_edge  = {"top_edge" : "CrossEntropyLoss", "res_edge" : "CrossEntropyLoss"}
                gnn.o_graph = {"ntops"    : "CrossEntropyLoss", "signal"   : "CrossEntropyLoss"}
                gnn.i_node  = ["pt", "eta", "phi", "energy", "charge"]
                gnn.i_graph = ["met", "phi"]
                gnn.device  = "cuda:" + str(x%2)
                gnn.checkpoint_path = mdl + kf[k] + "_model.pt"
                name = "ROOT/" + graph_name + graph_prefix + model_name + "/" + j + "/" + ep + "/" + kf[k]
                ana.AddModelInference(gnn, name)
                x+=1

    ana.Start()
    if len(selection_name):
        pathlib.Path(pth).mkdir(parents = True, exist_ok = True)
        f = open(pth + str(i) +".pkl", "wb")
        pickle.dump(selection_container[selection_name], f)
        f.close()
        print(i, len(ls))
    i+=1
    del ana
    if i > 40: break

ana = Analysis()
ana.Start()
method = plotting_method[study]
method.figures.figure_path = figure_path
method.figures.metacache = ana.GetMetaData
if study == "topefficiency": method.figures.TopEfficiency("./serialized-data/" + mrk + "/" + epoch + "/" + kfold + "/")

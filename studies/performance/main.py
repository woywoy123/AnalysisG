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
threads        = 12

graph_cache = "/scratch/tnom6927/graph-data-mc16-full/"
#root_model  = "/import/wu1/tnom6927/TrainingOutput/training-sessions/"
#data_path   = "/import/wu1/tnom6927/TrainingOutput/mc16-full/" # <----- cluster
#data_path = "/CERN/Samples/mc16-full" # <----- local
data_path = "/CERN/trainings/mc16-full-inference/ROOT/" + graph_name + graph_prefix + "Grift/MRK-1/epoch-18/kfold-1/"

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


ls = list(build_samples(data_path, "**/*.root", 5))
if fetch_meta:
    for i in ls:
        ana = Analysis()
        for k in i: ana.AddSamples(k, k.split("/")[-1])
        ana.FetchMeta = True
        ana.SumOfWeightsTreeName = "sumWeights"
        ana.Start()
    exit()

event_name     = "BSM4Tops"      if inference_mode     else "EventGNN"
graph_name     = graph_name      if inference_mode     else ""
model_name     = "Grift"         if inference_mode     else ""
selection_name = "TopEfficiency" if not inference_mode else ""

i = 0
for k in ls:
    if plot_only: break
    ana = Analysis()
    ana.Threads = threads
    ana.GraphCache = graph_cache
    if len(model_states): ana.FetchMeta = False

    try:
        selection_container[selection_name] = selection_method[selection_name]()
        ana.AddSelection(selection_container[selection_name])
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

    for j in model_states:
        for ep in model_states[j]:
            mdl = root_model + graph_name + graph_prefix + "/" + model_name + "/" + j + "/state/" + ep + "/"
            kf, dev = model_states[j][ep]
            for i in range(len(kf)):
                gnn = model_method[model_name]()
                gnn.o_edge  = {"top_edge" : "CrossEntropyLoss", "res_edge" : "CrossEntropyLoss"}
                gnn.o_graph = {"ntops"    : "CrossEntropyLoss", "signal"   : "CrossEntropyLoss"}
                gnn.i_node  = ["pt", "eta", "phi", "energy", "charge"]
                gnn.device  = dev[i]
                gnn.checkpoint_path = mdl + kf[i] + "_model.pt"
                name = "ROOT/" + graph_name + graph_prefix + "_" + model_name + "/" + j + "/" + ep + "/" + kf[i]
                ana.AddModelInference(gnn, name)
    ana.Start()
    if len(selection_name):
        f = open("./serialized-data/" + str(i) +".pkl", "wb")
        pickle.dump(selection_container[selection_name], f)
        f.close()
        print(i, len(ls))
        i+=1
    del ana

method = plotting_method[study]
method.figures.figure_path = figure_path
if study == "topefficiency": method.figures.TopEfficiency("./serialized-data/")

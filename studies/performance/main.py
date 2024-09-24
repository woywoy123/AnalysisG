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
model_method = {"RecursiveGraphNeuralNetwork" : RecursiveGraphNeuralNetwork, "Experimental" : Experimental}

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

graph_name  = "GraphJets"
inference_mode = False
plot_only      = True

#root_model = "/import/wu1/tnom6927/TrainingOutput/training-sessions/"
#data_path  = "/scratch/tnom6927/mc16-full/"
data_path = "ProjectName/ROOT/GraphJets_Experimental/MRK-1/epoch-9/kfold-1/"

model_states = {}
if inference_mode:
    model_states |= {
            "MRK-1" : {
                "epoch-1"  : (["kfold-1"], ["cuda:1"]),
                "epoch-74" : (["kfold-5"], ["cuda:0"]),
            }
    }

sample_path_ = {
        "singletop"  : data_path + "sorted-data/singletop"  + "/*",
        "ttH125"     : data_path + "sorted-data/ttH125"     + "/*",
        "ttbarHT1k"  : data_path + "sorted-data/ttbarHT1k"  + "/*",
        "SM4topsNLO" : data_path + "sorted-data/SM4topsNLO" + "/*",
        "ttbar"      : data_path + "sorted-data/ttbar"      + "/*",
        "ttbarHT1k5" : data_path + "sorted-data/ttbarHT1k5" + "/*",
        "ttbarHT6c"  : data_path + "sorted-data/ttbarHT6c"  + "/*",
        "Ztautau"    : data_path + "sorted-data/Ztautau"    + "/*",
        "llll"       : data_path + "sorted-data/llll"       + "/*",
        "lllv"       : data_path + "sorted-data/lllv"       + "/*",
        "llvv"       : data_path + "sorted-data/llvv"       + "/*",
        "lvvv"       : data_path + "sorted-data/lvvv"       + "/*",
        "tchan"      : data_path + "sorted-data/tchan"      + "/*",
        "tt"         : data_path + "sorted-data/tt"         + "/*",
        "ttee"       : data_path + "sorted-data/ttee"       + "/*",
        "ttmumu"     : data_path + "sorted-data/ttmumu"     + "/*",
        "tttautau"   : data_path + "sorted-data/tttautau"   + "/*",
        "ttW"        : data_path + "sorted-data/ttW"        + "/*",
        "ttZnunu"    : data_path + "sorted-data/ttZnunu"    + "/*",
        "ttZqq"      : data_path + "sorted-data/ttZqq"      + "/*",
        "tW"         : data_path + "sorted-data/tW"         + "/*",
        "tZ"         : data_path + "sorted-data/tZ"         + "/*",
        "Wenu"       : data_path + "sorted-data/Wenu"       + "/*",
        "WH125"      : data_path + "sorted-data/WH125"      + "/*",
        "WlvZqq"     : data_path + "sorted-data/WlvZqq"     + "/*",
        "Wmunu"      : data_path + "sorted-data/Wmunu"      + "/*",
        "WplvWmqq"   : data_path + "sorted-data/WplvWmqq"   + "/*",
        "WpqqWmlv"   : data_path + "sorted-data/WpqqWmlv"   + "/*",
        "WqqZll"     : data_path + "sorted-data/WqqZll"     + "/*",
        "WqqZvv"     : data_path + "sorted-data/WqqZvv"     + "/*",
        "Wt"         : data_path + "sorted-data/Wt"         + "/*",
        "Wtaunu"     : data_path + "sorted-data/Wtaunu"     + "/*",
        "Zee"        : data_path + "sorted-data/Zee"        + "/*",
        "ZH125"      : data_path + "sorted-data/ZH125"      + "/*",
        "Zmumu"      : data_path + "sorted-data/Zmumu"      + "/*",
        "ZqqZll"     : data_path + "sorted-data/ZqqZll"     + "/*",
        "ZqqZvv"     : data_path + "sorted-data/ZqqZvv"     + "/*",
}

if not inference_mode:
    ls = sum(list(build_samples(data_path, "./*/*.root", 10)), [])
    sample_path_ = {k.split("/")[-1] : k for k in ls}

event_name     = "BSM4Tops"      if inference_mode     else "EventGNN"
graph_name     = graph_name      if inference_mode     else ""
model_name     = "Experimental"  if inference_mode     else ""
selection_name = "TopEfficiency" if not inference_mode else ""

for k in build_samples(data_path + "sorted-data", "./*/*/*.root", 10):
    sample_path = {}
    for l in k:
        for t in sample_path_:
            if "/" + t + "/" not in l: continue
            if t not in sample_path: sample_path[t] = []
            sample_path[t] += [l]
            break

    skip = 0
    for l in sample_path:
        iox = IO(sample_path[l])
        iox.Trees = ["nominal"]
        iox.Leaves = ["weight_mc"]
        skip = len(iox)
    if skip == 0: continue
    if skip < 1000: th = 1
    else: th = 2

    ana = Analysis()
    ana.Threads = th
    ana.GraphCache = "ProjectName"
    if len(model_states): ana.FetchMeta = False
    for j in sample_path:
        for p in sample_path[j]: ana.AddSamples(p, j)

        try: ana.AddEvent(event_method[event_name](), j)
        except KeyError: pass
        except NameError: pass

        try: ana.AddGraph(graph_method[graph_name](), j)
        except KeyError: pass
        except NameError: pass

    for j in model_states:
        for ep in model_states[j]:
            mdl = root_model + graph_name + "/" + model_name + "/" + j + "/state/" + ep + "/"
            kf, dev = model_states[j][ep]
            for i in range(len(kf)):
                gnn = model_method[model_name]()
                gnn.o_edge = {"top_edge" : "CrossEntropyLoss", "res_edge" : "CrossEntropyLoss"}
                gnn.o_graph = {"ntops"   : "CrossEntropyLoss", "signal"   : "CrossEntropyLoss"}
                gnn.i_node = ["pt", "eta", "phi", "energy"]
                gnn.device = dev[i]
                gnn.checkpoint_path = mdl + kf[i] + "_model.pt"
                name = "ROOT/" + graph_name + "_" + model_name + "/" + j + "/" + ep + "/" + kf[i]
                ana.AddModelInference(gnn, name)
    ana.Start()
    del ana
    sample_path = {}

i = 0
for j, k in sample_path_.items():
    if plot_only: break
    ana = Analysis()
    ana.Threads = 12
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

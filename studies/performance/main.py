from AnalysisG.generators import Analysis

# event implementation
from AnalysisG.events.bsm_4tops import BSM4Tops
from AnalysisG.events.gnn import EventGNN

# study
from AnalysisG.selections.performance.topefficiency.topefficiency import TopEfficiency

# figures
import topefficiency
import pickle
import pathlib
import time
from pathlib import Path

study = "topefficiency"
smpls = "model-data/MRK-1/epoch-1/kfold-1/"
p = Path(smpls)
files = [str(x).replace(str(x).split("/")[-1], "") + "*" for x in p.glob("**/*.root") if str(x).endswith("root")]
files = list(set(files))

plotting_method = {
    "topefficiency" : topefficiency
}

gen_data = False
model_ev = True

figure_path = "./Output/"
mass = "None"
method = plotting_method[study]
method.figures.figure_path = figure_path
method.figures.mass_point  = "Mass." + mass + ".GeV"

for f in files:
    pth = f.split("/")[-2]
    Path("./serialized-data/").mkdir(parents = True, exist_ok = True)
    if not gen_data: continue
    ev = EventGNN() if model_ev else BSM4Tops()

    sel = None
    if study == "topefficiency": sel = TopEfficiency()

    ana = Analysis()
    ana.AddSamples(f, "tmp")
    ana.AddEvent(ev, "tmp")
    ana.AddSelection(sel)
    ana.Threads = 2
    ana.Start()
    time.sleep(1)

    f = open("./serialized-data/" + pth + ".pkl", "wb")
    pickle.dump(sel, f)
    f.close()

#f = open(study + "-" + mass_point + ".pkl", "rb")
#pres = pickle.load(f)
#f.close()

pres = "./serialized-data/"
print("plotting: " + study)
if study == "topefficiency": method.figures.TopEfficiency(pres)

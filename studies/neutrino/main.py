from AnalysisG.selections.neutrino.combinatorial import Combinatorial
from AnalysisG.selections.neutrino.validation import Validation

from AnalysisG.events.bsm_4tops import BSM4Tops
from AnalysisG import Analysis

from combinatorial.figures import double_neutrino, missing_energy
from validation.figures import nunuValidation

import pathlib
import pickle

i = 0
run = True
pth = "./data/"
data_path = "/home/tnom6927/Downloads/mc16/ttH_tttt_m400/*"

if run:
    data = Combinatorial() #Validation()
    ev = BSM4Tops()

    ana = Analysis()
    ana.Threads = 1
    ana.AddSelection(data)
    ana.AddEvent(ev, "nu")
    ana.AddSamples(data_path, "nu")
    ana.Start()

    pathlib.Path(pth).mkdir(parents = True, exist_ok = True)
    f = open(pth + str(i) +".pkl", "wb")
    pickle.dump(data, f)
    f.close()

data = pickle.load(open(pth+str(i) + ".pkl", "rb"))
#nunuValidation(data)
double_neutrino(data)

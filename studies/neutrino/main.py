from AnalysisG.selections.neutrino import Neutrino
from AnalysisG.events.bsm_4tops import BSM4Tops
from AnalysisG import Analysis

from figures import *
import pathlib
import pickle

i = 0
run = True
pth = "./data/"
data_path = "/CERN/Samples/mc16-full/ttH_tttt_m400/DAOD_TOPQ1.21955708._000011.root"

if run:
    data = Neutrino()
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

else: data = pickle.load(open(pth+str(i) + ".pkl", "rb"))

#missing_energy(data)
double_neutrino(data, True)

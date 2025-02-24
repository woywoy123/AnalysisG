from AnalysisG import Analysis
from AnalysisG.core.io import IO
from AnalysisG.events.ssml_mc20 import SSML_MC20
from AnalysisG.selections.analysis.regions.regions import Regions
from figures import entry

import pickle

smpls = "/home/tnom6927/Downloads/histo/ttZp.503575/user.rqian.42182125._000001.output.root"
ev = SSML_MC20()
sel = None # Regions()

if sel is not None:
    ana = Analysis()
    ana.AddSamples(smpls, "ttZp")
    ana.AddEvent(ev, "ttZp")
    ana.AddSelection(sel)
    ana.Threads = 1
    ana.FetchMeta = True
    ana.SumOfWeightsTreeName = "CutBookkeeper_*_NOSYS"
    ana.Start()

#smpl = IO(smpls)
#smpl.MetaCachePath = "./ProjectName/"
#smpl.EnablePyAMI = True
#smpl.Trees = ["reco"]
#smpl.Leaves = ["weight_mc_NOSYS"]
#smpl.SumOfWeightsTreeName = "CutBookkeeper_*_NOSYS:metadata"
#smpl.Keys

#meta = smpl.MetaData()
#meta = list(meta.values())[0]
#print(meta)

if sel is not None: pickle.dump(sel, open("pkl-data.pkl", "wb"))
try: sel = pickle.load(open("pkl-data.pkl", "rb"))
except: pickle.dump(sel, open("pkl-data.pkl", "wb"))
entry(sel)







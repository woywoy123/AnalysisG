from AnalysisG.core.plotting import *
from AnalysisG.generators import Analysis
from AnalysisG.selections.example.met.met import MET
from AnalysisG.events.bsm_4tops import BSM4Tops

evn = BSM4Tops()
sel = MET()

root = "/home/tnom6927/Downloads/mc16/"
x = Analysis()
x.AddSamples(root + "MadGraphPythia8EvtGen_noallhad_ttH_tttt_m1000/DAOD_TOPQ1.21955751._000019.root", "m1000")
x.AddEvent(evn, "m1000")
x.AddSelection(sel)
x.SumOfWeightsTreeName = "sumWeights"
x.FetchMeta = True
x.Start()

print(sel.missing_et)
print(sel.PassedWeights)
metx = sel.GetMetaData
print(list(metx.values())[0].expected_events())

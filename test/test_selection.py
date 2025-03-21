from AnalysisG import Analysis
from AnalysisG.selections.example.met.met import MET
from AnalysisG.events.bsm_4tops.event_bsm_4tops import BSM4Tops

root1 = "./samples/dilepton/*"
ev = BSM4Tops()
sl = MET()

ana = Analysis()
ana.AddSamples(root1, "tmp")
ana.AddEvent(ev, "tmp")
ana.AddSelection(sl)
ana.DebugMode = False
ana.SaveSelectionToROOT = True
ana.Start()


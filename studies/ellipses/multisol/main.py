from AnalysisG import Analysis
from AnalysisG.events.bsm_4tops import BSM4Tops
from AnalysisG.selections.neutrino.multisol.multisol import MultiSol
from conuic.main import run

smpl = "../../../test/samples/sample1/smpl1.root" #DAOD_TOPQ1.21955717._000001.root"

x = BSM4Tops()
s = MultiSol()

run(False)
ana = Analysis()
ana.AddSamples(smpl,"tmp")
ana.AddEvent(x, "tmp")
ana.AddSelection(s)
ana.Threads = 1
ana.DebugMode = True
ana.Start()




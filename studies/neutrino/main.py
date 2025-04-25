from AnalysisG import Analysis
from AnalysisG.events.bsm_4tops import BSM4Tops
from AnalysisG.selections.neutrino.validation.validation import Validation
from AnalysisG.selections.neutrino.combinatorial.combinatorial import NuNuCombinatorial
from validation.figures import validation
from combinatorial.figures import combinatorial

pax = "/home/tnom6927/Downloads/mc16/ttbar/user.tnommens.40945479._000013.output.root"
ev = BSM4Tops()
#sl = Validation()

sl = NuNuCombinatorial()
#ana = Analysis()
##ana.DebugMode = True
#ana.Threads = 1
#ana.SaveSelectionToROOT = True
#ana.AddSamples(pax, "ttbar")
#ana.AddEvent(ev, "ttbar")
#ana.AddSelection(sl)
#ana.Start()

sl.InterpretROOT("./ProjectName/Selections/" + sl.__name__() + "-" + ev.__name__() + "/ttbar/", "nominal")

for i in sl:
#    print(i.TruthTops)
    try: print(i.RecoTops["top_children"][0][0])
    except: pass

#validation(None)

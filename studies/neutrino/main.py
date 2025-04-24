from AnalysisG import Analysis
from AnalysisG.events.bsm_4tops import BSM4Tops
from AnalysisG.selections.neutrino.validation.validation import Validation
from figures import entry

pax = "/home/tnom6927/Downloads/mc16/ttbar/user.tnommens.40945479._000013.output.root"
ev = BSM4Tops()
sl = Validation()

#ana = Analysis()
#ana.Threads = 2
#ana.SaveSelectionToROOT = True
#ana.AddSamples(pax, "ttbar")
#ana.AddEvent(ev, "ttbar")
#ana.AddSelection(sl)
#ana.Start()

#sl.InterpretROOT("./ProjectName/Selections/" + sl.__name__() + "-" + ev.__name__() + "/user.tnommens.40945586._001002.output.root", "nominal")
entry(None)

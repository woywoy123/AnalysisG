from AnalysisG import Analysis
from AnalysisG.events.bsm_4tops import BSM4Tops
from AnalysisG.selections.neutrino.validation.validation import Validation


pax = "/home/tnom6927/Downloads/mc16/ttbar/user.tnommens.40945479._000013.output.root"


ev = BSM4Tops()
sl = Validation()
sl.InterpretROOT("./ProjectName/Selections/" + sl.__name__() + "-" + ev.__name__(), "nominal")

x = 0
evx = sl.Events
for i in evx:
    print("---------")
    print("->", i.TruthNeutrinos)
    print("+>", i.DynamicNeutrino)
    print("#>", i.StaticNeutrino)
    print(i.Particles)
    if x == 10: exit()
    x+=1

#ana = Analysis()
#ana.Threads = 2
#ana.SaveSelectionToROOT = True
#ana.AddSamples(pax, "ttbar")
#ana.AddEvent(ev, "ttbar")
#ana.AddSelection(sl)
#ana.Start()



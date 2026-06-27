from AnalysisG import Analysis
from AnalysisG.selections.performance.topefficiency.topefficiency import TopEfficiency
from AnalysisG.events.gnn import EventGNN
from figures import *

data_path = "/home/tnom6927/scratch/k-10/*"

name = "tttt"
run = True
sel = TopEfficiency()
ana = Analysis()

ev = EventGNN()
ana.Threads = 12
#ana.DebugMode = True
ana.AddEvent(ev, "k10")
ana.AddSamples(data_path, "k10")
ana.AddSelection(sel)
ana.SaveSelectionToROOT = True
ana.Start()
#sel.dump(name)

#sel = TopEfficiency()
#selttbar = sel.load("ttbar")
#
#sel = TopEfficiency()
#seltttt = sel.load("tttt")
#
#entry(selttbar, seltttt)




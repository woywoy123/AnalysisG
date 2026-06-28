from AnalysisG import Analysis
from AnalysisG.selections.performance.topefficiency.topefficiency import TopEfficiency
from AnalysisG.metrics.topefficiency.metric_topefficiency import TopEfficiencyMetric
from AnalysisG.events.gnn import EventGNN
from analysis import *




data_path = "/CERN/trainings/results/usyd/Grift-MRK-1/epoch-1/*"

name = "tttt"
run = True
#ev = EventGNN()

mx = TopEfficiencyMetric()
mx.InterpretROOT(
    "/home/tnom6927/scratch/Selections/topefficiency-gnn_event/Grift-MRK-1/", 
    [180], [10], "k-", "Grift-"
)

entry(mx)





#ana = Analysis()
#ana.Threads = 12
#ana.DebugMode = True
#ana.AddEvent(ev, "MRK-1.epoch-1")
#ana.AddSamples(data_path, "MRK-1.epoch-1")
#ana.AddSelection(sel)
#ana.SaveSelectionToROOT = True
#ana.Start()
#
#sel.dump(name)
#sel = TopEfficiency()
#selttbar = sel.load("ttbar")
#
#sel = TopEfficiency()
#seltttt = sel.load("tttt")
#
#entry(selttbar, seltttt)




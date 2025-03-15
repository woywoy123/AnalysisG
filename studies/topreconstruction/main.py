from AnalysisG import Analysis
from AnalysisG.selections.performance.topefficiency.topefficiency import TopEfficiency
from AnalysisG.events.gnn import EventGNN
from figures import *

data_ttbar = "/CERN/trainings/mc16-full-inference/ROOT/GraphJets_bn_1_Grift/MRK-1/epoch-130/kfold-1/user.tnommens.410465.aMcAtNloPy8EvtGen.DAOD_TOPQ1.e6762_a875_r10201_p4514.bsm4t_gnn_mc16d_output_root/user.tnommens.40945548._000147.output.root"
data_4top = "/CERN/trainings/mc16-full-inference/ROOT/GraphJets_bn_1_Grift/MRK-1/epoch-130/kfold-1/user.tnommens.412043.aMcAtNloPythia8EvtGen.DAOD_TOPQ1.e7101_a875_r10724_p4514.bsm4t_gnn_mc16e_output_root/user.tnommens.40946130._000009.output.root"
data_path = data_ttbar

name = "ttbar"
run = False
sel = TopEfficiency()
ana = Analysis()

if run:
    ev = EventGNN()
    ana.Threads = 4
    ana.AddEvent(ev, "tmp")
    ana.AddSelection(sel)
    ana.AddSamples(data_path, "tmp")
ana.Start()
if run:
    sel.dump(name)
    exit()

sel = TopEfficiency()
selttbar = sel.load("ttbar")

sel = TopEfficiency()
seltttt = sel.load("tttt")

entry(selttbar, seltttt, ana.GetMetaData)




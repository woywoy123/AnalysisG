from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Plotting import TH1F
from Event import EventExperimental
#from AnalysisTopGNN.Deprecated import Event

Ana = Analysis()
Direc = "/home/tnom6927/Downloads/ttH_tttt_m1000/output.root"
Ana.InputSample("BSM4tops", Direc)
Ana.EventCache = True
Ana.DumpPickle = False
Ana.EventStop = 10
Ana.chnk = 1
Ana.Threads = 1
Ana.Event = EventExperimental
Ana.Launch()


Top = []
TopChildren = []
TopTruJetParton = []
TopTruthJet = []
TopJetParton = []
TopJet = []
for event in Ana:
    collect = {}
    event = event.Trees["nominal"]
    print(event.Tops)


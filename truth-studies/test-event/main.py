from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Plotting import TH1F
#from Event import EventExperimental
#from AnalysisTopGNN.Deprecated import Event
from AnalysisTopGNN.Events import Event



Ana = Analysis()
Direc = "/CERN/Samples/Dilepton/ttH_tttt_m1000/DAOD_TOPQ1.21955751._000018.root"
Ana.InputSample("BSM4tops", Direc)
Ana.EventCache = True
Ana.DumpPickle = False
#Ana.EventStop = 10
#Ana.EventStart = 434
Ana.chnk = 1
Ana.Threads = 1
Ana.Event = Event #Experimental
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
    print(event.DetectorObjects)


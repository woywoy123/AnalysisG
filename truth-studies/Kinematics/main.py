from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Events import Event
from TruthTops import *
from TruthChildren import *
from TruthMatching import *

direc = "/home/tnom6927/Downloads/CustomAnalysisTopOutputTest/tttt/QU_0.root"
Ana = Analysis()
Ana.InputSample("tttt", direc)
Ana.Event = Event
#Ana.EventStop = 100
Ana.EventCache = False
Ana.DumpPickle = True
Ana.Launch()

TruthTopsAll(Ana)
TruthTopsHadron(Ana)
TruthChildrenAll(Ana)
TruthChildrenHadron(Ana)
TruthJetAll(Ana)

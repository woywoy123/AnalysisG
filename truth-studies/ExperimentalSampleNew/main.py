from AnalysisTopGNN.Generators import Analysis
from AnalysisTopGNN.Plotting import TH1F
from Event import *

Ana = Analysis()
Direc = "/home/tnom6927/Downloads/CustomAnalysisTopOutputTest/output.root"
Ana.InputSample("ttbar", Direc)
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
    tops = event.Tops
    
    for t in tops:
        al = t.Children
        
        trujet = t.TruthJets
        trujetpart = [jt for k in al for jt in k.TruthJetPartons]
        
        jets = t.Jets
        jetpart = [jt for k in al for jt in k.JetPartons]

        Top.append(t.CalculateMass())
        TopChildren.append(sum(al).CalculateMass())
        
        TopTruthJet.append(sum(trujet).CalculateMass())
        TopTruJetParton.append(sum(trujetpart).CalculateMass())
       
        if len(jets) != 0:
            TopJet.append(sum(jets).CalculateMass())
        if len(jetpart) != 0:
            TopJetParton.append(sum(jetpart).CalculateMass())



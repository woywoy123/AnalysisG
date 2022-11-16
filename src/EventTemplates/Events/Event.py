from AnalysisTopGNN.Templates import EventTemplate
from Particles import *

class Event(EventTemplate):

    def __init__(self):
        EventTemplate.__init__(self)
        self.Objects = {
                "Tops" : Top(), 
                "Children" : Children(),
                "TruthJets" : TruthJet(), 
                "TruthJetPartons" : TruthJetPartons(),
                "Jets" : Jets(), 
                "JetPartons" : JetPartons(),
        }
       
        self.Trees = ["nominal"]

        self.Lumi = "weight_mc"
        self.mu = "mu"

        self.DefineObjects()

    def CompileEvent(self):
        
        for i in self.Children:
            t = self.Children[i]
            self.Tops[t.index].Children.append(t)

       
        for jp in self.TruthJetPartons:
            tjp = self.TruthJetPartons[jp]
            self.Children[tjp.index].TruthJetPartons.append(tjp)
            tjp.TruthJet.append(self.TruthJets[tjp.TruJetIndex])

        for jp in self.JetPartons:
            tjp = self.JetPartons[jp]
            self.Children[tjp.index].JetPartons.append(tjp)
            tjp.Jet.append(self.Jets[tjp.JetIndex])

        for i in self.TruthJets:
            for ti in self.TruthJets[i].index:
                if ti == -1:
                    continue
                self.Tops[ti].TruthJets.append(self.TruthJets[i])

        for i in self.Jets:
            for ti in self.Jets[i].index:
                if ti == -1:
                    continue
                self.Tops[ti].Jets.append(self.Jets[i])

        self.Tops = list(self.Tops.values())
        self.Children = list(self.Children.values())
        self.Jets = list(self.Jets.values())
        self.TruthJets = list(self.TruthJets.values())

from AnalysisTopGNN.Templates import EventTemplate
from AnalysisTopGNN.Particles.Particles import *

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
        
        self._Deprecated = False
        self._CommitHash = "master@e633cf7b9b51de362222bd3175bfec8e6c00026f"

    def CompileEvent(self):
        self.JetPartons = { i : self.JetPartons[i] for i in self.JetPartons if self.JetPartons[i].index != []}
        self.TruthJetPartons = { i : self.TruthJetPartons[i] for i in self.TruthJetPartons if self.TruthJetPartons[i].index != []}

        for i in self.Children:
            t = self.Children[i]
            if isinstance(t.index, list):
                continue
            self.Tops[t.index].Children.append(t)
       
        for jp in self.TruthJetPartons:
            tjp = self.TruthJetPartons[jp]
            tjp.TruthJet.append(self.TruthJets[tjp.TruJetIndex])
            self.TruthJets[tjp.TruJetIndex].Partons.append(tjp)

        for jp in self.JetPartons:
            tjp = self.JetPartons[jp]
            tjp.Jet.append(self.Jets[tjp.JetIndex])
            self.Jets[tjp.JetIndex].Partons.append(tjp)

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

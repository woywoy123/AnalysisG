from AnalysisTopGNN.Templates import EventTemplate
from Particles import *

class EventExperimental(EventTemplate):

    def __init__(self):
        EventTemplate.__init__(self)
        self.Objects = {
                "Tops" : Top(), 
                "Children" : Children(),
                "TruthJets" : TruthJet(),
                "TruthJetPartons" : TruthJetParton(),
                "Jets" : Jet(), 
                "JetPartons" : JetParton(),
        }
       
        self.Trees = ["nominal"]

        self.Lumi = "weight_mc"
        self.mu = "mu"

        self.DefineObjects()

    def CompileEvent(self):
        self.Tops = {t.index : t for t in self.Tops.values()} 
        self.Children = {c.index : c for c in self.Children.values()}
        self.TruthJets = {tj.index : tj for tj in self.TruthJets.values()}
        self.TruthJetPartons = {tj.index : tj for tj in self.TruthJetPartons.values()}
        self.Jets = {j.index : j for j in self.Jets.values()}
        self.JetPartons = {j.index : j for j in self.JetPartons.values()}
        
        for c in self.Children.values():
            self.Tops[c.TopIndex].Children.append(c)
        
        for tj in self.TruthJets.values():
            for ti in tj.TopIndex:
                if ti == -1:
                    continue
                tj.Tops.append(self.Tops[ti])
                self.Tops[ti].TruthJets.append(tj)
        
        for tjp in self.TruthJetPartons.values():
            self.TruthJets[tjp.TruthJetIndex].Parton.append(tjp)
            tjp.Children.append(self.TruthJets[tjp.TruthJetIndex])
            for ci in tjp.TopChildIndex:
                tjp.Parent.append(self.Children[ci])
 
        for j in self.Jets.values():
            for ti in j.TopIndex:
                if ti == -1:
                    continue
                j.Tops.append(self.Tops[ti])
                self.Tops[ti].Jets.append(j)
        
        for jp in self.JetPartons.values():
            self.Jets[jp.JetIndex].Parton.append(jp)
            jp.Children.append(self.Jets[jp.JetIndex])
            for ci in jp.TopChildIndex:
                jp.Parent.append(self.Children[ci])

        self.Tops = list(self.Tops.values())
        self.Children = list(self.Children.values())
        self.TruthJets = list(self.TruthJets.values())
        self.Jets = list(self.Jets.values())

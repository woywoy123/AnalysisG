from AnalysisTopGNN.Tools.Variables import VariableManager
import copy
import math
import torch
import LorentzVector

class ParticleTemplate(VariableManager):
    def __init__(self):
        self.Index = -1
        VariableManager.__init__(self)
        self.Parent = []
        self.Children = []

    def DeltaR(self, P):
        return math.sqrt(math.pow(P.eta-self.eta, 2) + math.pow(P.phi-self.phi, 2)) 

    def CalculateMass(self, lists = None, Name = "Mass"):
        if lists == None:
            lists = [self]
        v = torch.zeros((1, 4))
        for i in lists:
            v += LorentzVector.ToPxPyPzE(i.pt, i.eta, i.phi, i.e, "cpu")
        m = float(LorentzVector.MassFromPxPyPzE(v))
        setattr(self, Name + "_MeV", m)
        setattr(self, Name + "_GeV", m / 1000)

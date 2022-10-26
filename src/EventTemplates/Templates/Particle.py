from .Manager import VariableManager
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
        return m/1000
    
    def __del__(self):
        for i in self.__dict__:
            val = self.__dict__[i]
            del val, i
       
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)


    def __add__(self, other):
        
        v = torch.zeros((1, 4))
        v += LorentzVector.ToPxPyPzE(self.pt, self.eta, self.phi, self.e, "cpu")
        v += LorentzVector.ToPxPyPzE(other.pt, other.eta, other.phi, other.e, "cpu") 
        pmu = LorentzVector.TensorToPtEtaPhiE(v)
        pmu = pmu.tolist()[0]
        
        particle = ParticleTemplate() 
        setattr(particle, "pt", pmu[0])
        setattr(particle, "eta", pmu[1])
        setattr(particle, "phi", pmu[2])
        setattr(particle, "e", pmu[3])
        particle.Children += self.Children
        particle.Children += [p for p in other.Children if p not in particle.Children]
        
        return particle

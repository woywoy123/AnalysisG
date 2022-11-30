from .Manager import VariableManager
from AnalysisTopGNN.Vectors import *

class ParticleTemplate(VariableManager):
    def __init__(self):
        self.Index = -1
        VariableManager.__init__(self)
        self.Parent = []
        self.Children = []

    def DeltaR(self, P):
        return deltaR(P.eta, self.eta, P.phi, self.phi)

    def CalculateMass(self, lists = None, Name = "Mass"):
        if lists == None:
            lists = [self]

        v = [0, 0, 0, 0]
        for i in lists:
            v[0] += Px(i.pt, i.phi)
            v[1] += Py(i.pt, i.phi)
            v[2] += Py(i.pt, i.eta)
            v[3] += i.e
        m = PxPyPzEMass(v[0], v[1], v[2], v[3])
        self.__dict__[Name + "_MeV"] =  m
        self.__dict__[Name + "_GeV"] = m / 1000
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
        
        x = Px(self.pt, self.phi) + Px(other.pt, other.phi)
        y = Py(self.pt, self.phi) + Py(other.pt, other.phi)
        z = Pz(self.pt, self.eta) + Pz(other.pt, other.eta)
        e = self.e + other.e

        particle = ParticleTemplate() 
        particle.__dict__["pt"] = PT(x, y)
        particle.__dict__["eta"] = Eta(x, y, z)
        particle.__dict__["phi"] = Phi(x, y)
        particle.__dict__["e"] = e
        particle.Children += self.Children
        particle.Children += [p for p in other.Children if p not in particle.Children]
        
        return particle
    
    def DecayLeptonically(self):
        return True if sum([1 for k in self.Children if abs(k.pdgid) in [11, 12, 13, 14, 15, 16]]) > 0 else False

    def __str__(self, caller = False):
        PDGID = { 
        1 : "d"        ,  2 : "u"             ,  3 : "s", 
                 4 : "c"        ,  5 : "b"             , 11 : "e", 
                 12 : "$\\nu_e$" , 13 : "$\mu$"         , 14 : "$\\nu_{\mu}$", 
                 15 : "$\\tau$"  , 16 : "$\\nu_{\\tau}$", 21 : "g", 
                 22 : "$\\gamma$"}
        
        string = ""
        if "pdgid" in self.__dict__:
            string += "pdgid: " + str(self.pdgid)
            string += " Symbol " + PDGID[self.pdgid] + " "
        string += "eta: " + str(self.eta)
        string += "phi: " + str(self.phi)
        string += "pt: " + str(self.pt)
        
        if caller:
            return string

        for i in self.Children:
            string += " -> " + str(i, True)
        return string

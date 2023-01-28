import torch
from .Manager import VariableManager
from LorentzVector import *

class ParticleTemplate(VariableManager):
    def __init__(self):
        self.index = -1
        VariableManager.__init__(self)
        self.Parent = []
        self.Children = []

    def DeltaR(self, P):
        t1 = torch.tensor([[P.pt, P.eta, P.phi, P.e]])
        t2 = torch.tensor([[self.pt, self.eta, self.phi, self.e]])
        dr = TensorDeltaR(t1, t2).tolist()[0][0]
        return dr

    @property
    def Mass(self):
        t = torch.tensor([self.pt, self.eta, self.phi, self.e])
        return MassFromPtEtaPhiE(t).tolist()[0][0] / 1000
    
    def __del__(self):
        for i in self.__dict__:
            val = self.__dict__[i]
            del val, i

    def __radd__(self, other):
        if other == 0:
            p = ParticleTemplate()
            p.__dict__["pt"] = self.pt
            p.__dict__["eta"] = self.eta
            p.__dict__["phi"] = self.phi
            p.__dict__["e"] = self.e
            return p
        else:
            return self.__add__(other)

    def __add__(self, other):

        P1 = ToPxPyPzE(self.pt, self.eta, self.phi, self.e, "cpu")
        P2 = ToPxPyPzE(other.pt, other.eta, other.phi, other.e, "cpu")
        Pmu = TensorToPtEtaPhiE(P1 + P2).tolist()[0]

        particle = ParticleTemplate()
        particle.__dict__["pt"] = Pmu[0]
        particle.__dict__["eta"] = Pmu[1]
        particle.__dict__["phi"] = Pmu[2]
        particle.__dict__["e"] = Pmu[3]
        particle.Children += self.Children
        particle.Children += [p for p in other.Children if p not in particle.Children]

        return particle

    def DecayLeptonically(self):
        return True if sum([1 for k in self.Children if abs(k.pdgid) in [11, 12, 13, 14, 15, 16]]) > 0 else False

    def __str__(self, caller = False):
        PDGID = {
                  1 : "d"           ,  2 : "u"             ,  3 : "s",
                  4 : "c"           ,  5 : "b"             ,  6 : "t",
                 11 : "e"           , 12 : "$\\nu_e$"      , 13 : "$\mu$",
                 14 : "$\\nu_{\mu}$", 15 : "$\\tau$"       , 16 : "$\\nu_{\\tau}$",
                 21 : "g"           , 22 : "$\\gamma$"}

        string = ""
        if "pdgid" in self.__dict__:
            string += "======== "
            string += "pdgid: " + str(self.pdgid) + " "
            string += "Symbol: " + PDGID[abs(self.pdgid)] + " ====\n"
        string += "eta: " + str(self.eta) + "\n"
        string += "phi: " + str(self.phi) + "\n"
        string += "pt: " + str(self.pt) + "\n"

        if caller:
            return string

        for i in self.Children:
            string += " -> " + i.__str__(caller = True)
        return string

    @property
    def is_lep(self):
        return False

    @property
    def is_nu(self):
        return False

    @property
    def is_b(self):
        return False

    @property
    def is_add(self):
        return not (self.is_lep or self.is_nu or self.is_b)

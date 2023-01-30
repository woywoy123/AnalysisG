from .Manager import VariableManager
from AnalysisTopGNN.Vectors import * 

class ParticleTemplate(VariableManager):
    def __init__(self):
        self.index = -1
        VariableManager.__init__(self)
        self.Parent = []
        self.Children = []

    def DeltaR(self, P):
        return deltaR(P.eta, self.eta, P.phi, self.phi)
    
    @property
    def Mass(self):
        return PxPyPzEMass(self._px, self._py, self._pz, self.e)

    def _istype(self, lst):
        if "pdgid" in self.__dict__:
            return abs(self.pdgid) in lst
        return False

    @property
    def is_lep(self):
        return self._istype([11, 13, 15])

    @property
    def is_nu(self):
        return self._istype([12, 14])

    @property
    def is_b(self):
        return self._istype([5])

    @property
    def is_add(self):
        return not (self.is_lep or self.is_nu or self.is_b)

    @property
    def _px(self):
        return Px(self.pt, self.phi)

    @property
    def _py(self):
        return Py(self.pt, self.phi)

    @property
    def _pz(self):
        return Pz(self.pt, self.eta)

    @property
    def _eta(self):
        if len([1 for i in ["px", "py", "pz"] if i in self.__dict__]) == 3:
            return Eta(self.px, self.py, self.pz)
        return self.eta

    @property
    def _phi(self):
        if len([1 for i in ["px", "py"] if i in self.__dict__]) == 2:
            return Phi(self.px, self.py)
        return self.phi

    @property
    def _pt(self):
        if len([1 for i in ["px", "py"] if i in self.__dict__]) == 2:
            return PT(self.px, self.py)
        return self.pt
    
    @property
    def LeptonicDecay(self):
        return sum([1 for k in self.Children if abs(k.pdgid) in [11, 12, 13, 14, 15, 16]]) > 0

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


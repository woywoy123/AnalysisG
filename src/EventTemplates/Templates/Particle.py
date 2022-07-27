from AnalysisTopGNN.Tools.Variables import VariableManager
import copy
import math
import torch
import LorentzVector

class ParticleTemplate(VariableManager):
    def __init__(self):
        self.Index = -1
        VariableManager.__init__(self)
    
    def _DefineParticle(self):
        self.CompileKeyMap()
        self.ListAttributes()
   
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

    def CalculateMassFromChildren(self):
        if len(self.Decay) != 0:
            self.CalculateMass(self.Decay, "Mass")
        if len(self.Decay_init) != 0:    
            self.CalculateMass(self.Decay_init, "Mass_init")


class CompileParticles:
    def __init__(self, Dictionary, Particles):
        self.__Dictionary = Dictionary
        self.__len = -1
        self.__Part = Particles
        self.__Keys = {}
        
        for key in self.__Dictionary:
            val = self.__Dictionary[key]
            if key in self.__Part.KeyMap:
                self.__Keys[self.__Part.KeyMap[key]] = val
                self.__len = len(val)

    def Compile(self, ClearVal = True):
        Output = {}
        
        for i in range(self.__len):
            Output[i] = []
            for k in self.__Keys:
                val = self.__Keys[k][i]
                try:
                    __sub = len(val)
                    if len(Output[i]) == 0:
                        for j in range(__sub):
                            P = copy.deepcopy(self.__Part)
                            P.Index = i
                            Output[i].append(P)
                    for j in range(__sub):
                        Output[i][j].SetAttribute(k, val[j]) 
                except:
                    if len(Output[i]) == 0:
                        P = copy.deepcopy(self.__Part)
                        P.Index = i
                        Output[i].append(P)
                    Output[i][0].SetAttribute(k, val)
        
        for i in Output:
            for p in Output[i]:
                if ClearVal:
                    p.ClearVariable()
        del self.__Part
        del self.__Keys
        del self.__Dictionary
        del self.__len
        return Output
 

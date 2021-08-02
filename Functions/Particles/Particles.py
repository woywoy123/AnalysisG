from Functions.Tools.Variables import VariableManager
from Functions.Tools.DataTypes import DataTypeCheck
import copy
import math
from skhep.math.vectors import LorentzVector
from ROOT import Math

class Particle(VariableManager, DataTypeCheck):
    def __init__(self):
        VariableManager.__init__(self)
        DataTypeCheck.__init__(self)
        self.pt = self.Type + "_pt"
        self.eta = self.Type + "_eta"
        self.phi = self.Type + "_phi"
        self.e = self.Type + "_e"
        self.Index = -1
        self.Signal = -1
        self.CompileKeyMap()
        self.ListAttributes()
        self.Decay_init = []
        self.Decay = []
        self.ParentPDGID = ""

    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def DeltaR(self, P):
        return math.sqrt(math.pow(P.eta-self.eta, 2) + math.pow(P.phi-self.phi, 2)) 

    def CalculateMass(self):
        v = LorentzVector()
        v.setptetaphie(self.pt, self.eta, self.phi, self.e)
        self.Mass = v.mass

class Lepton(Particle):
    def __init__(self):
        self.charge = self.Type + "_charge"
        self.topoetcone20 = self.Type + "_topoetcone20"
        self.d0sig = self.Type + "_d0sig"
        self.delta_z0_sintheta = self.Type + "_delta_z0_sintheta"
        self.true_type = self.Type + "_true_type"
        self.true_origin = self.Type + "_true_origin"
        self.true_IFFclass = self.Type + "_true_IFFclass"
        self.true_isPrompt = self.Type + "_true_isPrompt"

        Particle.__init__(self)

class Electron(Lepton):
    def __init__(self):
        self.Type = "el"
        self.ptvarcone20 = self.Type + "_ptvarcone20"
        self.CL = self.Type + "_CF"
        self.true_firstEgMotherTruthType = self.Type + "_true_firstEgMotherTruthType"
        self.true_firstEgMotherTruthOrigin = self.Type + "_true_firstEgMotherTruthOrigin"
        self.true_firstEgMotherPdgId = self.Type + "_true_firstEgMotherPdgId"
        self.true_isChargeFl = self.Type + "_true_isChargeFl"

        Lepton.__init__(self)

class Muon(Lepton):
    def __init__(self):
        self.Type = "mu"
        self.ptvarcone30 = self.Type + "_ptvarcone30"
        Lepton.__init__(self)

class TruthJet(Particle):
    def __init__(self):
        self.Type = "truthjet"
        self.flavour = self.Type + "_flavour"
        self.flavour_extended = self.Type + "_flavour_extended"
        self.nCHad = self.Type + "_nCHad"
        self.nBHad = self.Type + "_nBHad"
        Particle.__init__(self)

class Jet(Particle):
    def __init__(self):
        self.Type = "jet"
        self.jvt = self.Type + "_jvt"
        self.truthflav = self.Type + "_truthflav"
        self.truthPartonLabel = self.Type + "_truthPartonLabel"
        self.isTrueHS = self.Type + "_isTrueHS"
        self.truthflavExtended = self.Type + "_truthflavExtended"
        self.isbtagged_DL1r_77 = self.Type + "_isbtagged_DL1r_77"
        self.isbtagged_DL1r_70 = self.Type + "_isbtagged_DL1r_70"
        self.isbtagged_DL1r_60 = self.Type + "_isbtagged_DL1r_60"
        self.isbtagged_DL1r_85 = self.Type + "_isbtagged_DL1r_85"
        self.DL1r = self.Type + "_DL1r"
        self.DL1r_pb = self.Type + "_DL1r_pb"
        self.DL1r_pc = self.Type + "_DL1r_pc"
        self.DL1r_pu = self.Type + "_DL1r_pu"

        self.isbtagged_DL1_77 = self.Type + "_isbtagged_DL1_77"
        self.isbtagged_DL1_70 = self.Type + "_isbtagged_DL1_70"
        self.isbtagged_DL1_60 = self.Type + "_isbtagged_DL1_60"
        self.isbtagged_DL1_85 = self.Type + "_isbtagged_DL1_85"
        self.DL1 = self.Type + "_DL1"
        self.DL1_pb = self.Type + "_DL1_pb"
        self.DL1_pc = self.Type + "_DL1_pc"
        self.DL1_pu = self.Type + "_DL1_pu"

        Particle.__init__(self)

class Top(Particle):
    def __init__(self):
        self.Type = "truth_top"
        self.FromRes = "top_FromRes"
        self.charge = self.Type + "_charge"
        Particle.__init__(self)

class Truth_Top_Child(Particle):
    def __init__(self):
        self.Type = "truth_top_child"
        self.pdgid = self.Type + "_pdgid"
        Particle.__init__(self)

class Truth_Top_Child_Init(Particle):
    def __init__(self):
        self.Type = "truth_top_initialState_child"
        self.pdgid = "top_initialState_child_pdgid"
        Particle.__init__(self)

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

    def Compile(self):
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
        return Output
    

                 
               
                


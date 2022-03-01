from Functions.Tools.Variables import VariableManager
import copy
import math
from skhep.math.vectors import LorentzVector

class Particle(VariableManager):
    def __init__(self, Type = False):
        if Type != False:
            self.Type = "Particle"

        VariableManager.__init__(self)
        self.pt = self.Type + "_pt"
        self.eta = self.Type + "_eta"
        self.phi = self.Type + "_phi"
        self.e = self.Type + "_e"
        self.Index = -1

        if hasattr(self, "FromRes"):
            self.Signal = self.FromRes
        else:
            self.Signal = 0

        self.CompileKeyMap()
        self.ListAttributes()
        self.Decay_init = []
        self.Decay = []
    
    def __eq__(self, other):
        if (isinstance(other, Particle)):
            if self.__dict__ != self.__dict__:
                return False

            for k in self.__dict__:
                if getattr(self, k) != getattr(other, k):
                    return False
            return True
        else:
            return False

    def DeltaR(self, P):
        return math.sqrt(math.pow(P.eta-self.eta, 2) + math.pow(P.phi-self.phi, 2)) 

    def EnergyUnits(self, mass, name):
        setattr(self, name + "_eV", mass * 1000)
        setattr(self, name + "_MeV", mass)
        setattr(self, name + "_GeV", mass / 1000)

    def CalculateMass(self):
        v = LorentzVector()
        v.setptetaphie(self.pt, self.eta, self.phi, self.e)
        self.EnergyUnits(v.mass, "Mass")

    def CalculateMassFromChildren(self):
        v = self.CalculateVector(self.Decay)
        v_init = self.CalculateVector(self.Decay_init)
        self.EnergyUnits(v.mass, "Mass")
        self.EnergyUnits(v_init.mass, "Mass_init")

    def CalculateVector(self, lists):
        vec = LorentzVector()
        for i in lists:
            v = LorentzVector()
            v.setptetaphie(i.pt, i.eta, i.phi, i.e)
            vec += v
        return vec
    
    def PropagateSignalLabel(self):
        for i in self.Decay:
            i.Signal = self.Signal
            i.PropagateSignalLabel()

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
    


####### Default particles in ROOT files ###############

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

class RCSubJet(Particle):
    def __init__(self):
        self.Type="rcjetsub"
        Particle.__init__(self)
        self.Flav = None

class RCJet(Particle):
    def __init__(self):
        self.Type = "rcjet"
        self.d12 = self.Type + "_d12"
        self.d23 = self.Type + "_d23"
        Particle.__init__(self)
        self.Constituents = []

    def PropagateJetSignal(self):
        # 0: No Signal 
        # 1: All Signal 
        # 2: Contains at least one signal
        
        s = 0
        for i in self.Constituents:
            if i.Signal == 0:
                s += i.Signal
            if i.Signal == 1:
                s += i.Signal
        if s == len(self.Constituents) and len(self.Constituents) != 0:
            self.Signal = 1
        elif s == 0:
            self.Signal = 0
        elif s > 0:
            self.Signal = 2

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


################## CUSTOMIZED PARTICLES CLASSES ################

class Jet_C(Particle):
    def __init__(self):
        self.Type = "jet"
        self.truthflav = self.Type + "_truthflav"
        self.truthPartonLabel = self.Type + "_truthPartonLabel"
        self.truthflavExtended = self.Type + "_truthflavExtended"
        self.JetMapGhost = self.Type + "_map_Ghost"

        Particle.__init__(self)

class TruthJet_C(Particle):
    def __init__(self):
        self.Type = "truthjet"
        self.pdgid = self.Type + "_pdgid"
        self.GhostTruthJetMap = "GhostTruthJetMap"
        Particle.__init__(self)

class TruthTop_C(Particle):
    def __init__(self):
        self.Type = "truth_top"
        self.FromRes = self.Type + "_FromRes"
        Particle.__init__(self)

class TruthTopChild_C(Particle):
    def __init__(self):
        self.Type = "truth_top_child"
        self.charge = self.Type + "_charge"
        self.pdgid = self.Type + "_pdgid"
        Particle.__init__(self)

class TopPreFSR_C(Particle):
    def __init__(self):
        self.Type = "topPreFSR"
        self.charge = self.Type + "_charge"
        self.FromRes = "Gtop_FromRes"
        Particle.__init__(self)

class TopPostFSR_C(Particle):
    def __init__(self):
        self.Type = "topPostFSR"
        self.charge = self.Type + "_charge"
        self.FromRes = "Gtop_FromRes"
        Particle.__init__(self)

class TopPostFSRChildren_C(Particle):
    def __init__(self):
        self.Type = "topPostFSRchildren"
        self.charge = self.Type + "_charge"
        self.pdgid = self.Type + "_pdgid"
        Particle.__init__(self)

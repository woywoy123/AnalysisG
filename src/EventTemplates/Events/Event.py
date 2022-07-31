from AnalysisTopGNN.Particles.Particles import *
from AnalysisTopGNN.Templates import EventTemplate

class Event(EventTemplate):
    def __init__(self):
        EventTemplate.__init__(self)
        self.Objects = {
                            "Electrons" : Electron(), 
                            "Muons" : Muon(), 
                            "Jets" : Jet(), 
                            "TruthJets": TruthJet(), 
                            "TruthTops" : TruthTop(), 
                            "TruthTopChildren": TruthTopChildren(), 
                            "TopPreFSR" : TopPreFSR(),
                            "TopPostFSR" : TopPostFSR(),
                            "TopPostFSRChildren" : TopPostFSRChildren()
                        }
        self.Tree = ["nominal"]
        self.runNumber = "runNumber"
        self.eventNumber = "eventNumber"
        
        tag = "weight"
        self.pileup = tag + "_pileup"
        self.leptonSF = tag + "_leptonSF"
        self.globalLeptonTriggerSF = tag + "_globalLeptonTriggerSF"

        self.bTagSF_DL1_85 = tag + "_bTagSF_DL1_85"
        self.bTagSF_DL1_77 = tag + "_bTagSF_DL1_77"
        self.bTagSF_DL1_70 = tag + "_bTagSF_DL1_70"
        self.bTagSF_DL1_60 = tag + "_bTagSF_DL1_60"

        self.bTagSF_DL1r_85 = tag + "_bTagSF_DL1r_85"
        self.bTagSF_DL1r_77 = tag + "_bTagSF_DL1r_77"
        self.bTagSF_DL1r_70 = tag + "_bTagSF_DL1r_70"
        self.bTagSF_DL1r_60 = tag + "_bTagSF_DL1r_60"

        self.jvt = tag + "_jvt"
        self.indiv_SF_EL_Reco = tag + "_indiv_SF_EL_Reco"
        self.indiv_SF_EL_ID = tag + "_indiv_SF_EL_ID"
        self.indiv_SF_EL_Isol = tag + "_indiv_SF_EL_Isol"

        self.indiv_SF_MU_ID = tag + "_indiv_SF_MU_ID"
        self.indiv_SF_MU_Isol = tag + "_indiv_SF_MU_Isol"

        self.mu = "mu"
        self.mu_actual = "mu_actual"

        self.met = "met_met"
        self.met_phi = "met_phi" 

        self.DefineObjects()
        self.iter = -1

    def CompileEvent(self, ClearVal = True):
        self.CompileParticles(ClearVal)
        def FixList(Input):
            try:
                return [int(Input)]
            except:
                return list(Input)

        def RecursiveSignal(DecayList, sig, index):
            for p_i in DecayList:
                setattr(p_i, "FromRes", sig)
                setattr(p_i, "Index", index)
                RecursiveSignal(p_i.Decay_init, sig, index)
                RecursiveSignal(p_i.Decay, sig, index)
      
        for i in self.TruthTops:
            self.TruthTops[i][0].Decay_init += self.TruthTopChildren[i]

        for i in self.TopPostFSR:
            self.TopPostFSR[i][0].Decay_init += self.TopPostFSRChildren[i]

        for i in self.TruthJets:
            self.TruthJets[i][0].GhostTruthJetMap = FixList(self.TruthJets[i][0].GhostTruthJetMap)
            self.TruthJets[i][0].Index = -1
            self.TruthJets[i][0].FromRes = 0

        for i in self.Jets:
            self.Jets[i][0].JetMapGhost = FixList(self.Jets[i][0].JetMapGhost)
            self.Jets[i][0].JetMapTops = FixList(self.Jets[i][0].JetMapTops)

        for i in self.TruthJets:
            for t in self.TruthJets[i][0].GhostTruthJetMap:
                if t == -1:
                    continue
                self.TopPostFSR[t][0].Decay += self.TruthJets[i]

        for i in self.Jets:
            for tj in self.Jets[i][0].JetMapGhost:
                if tj == -1:
                    continue
                self.TruthJets[tj][0].Decay += self.Jets[i]

        self.Electrons = self.DictToList(self.Electrons)
        self.Muons = self.DictToList(self.Muons)
        self.Jets = self.DictToList(self.Jets)
        
        self.Leptons = []
        self.Leptons += self.Electrons
        self.Leptons += self.Muons
        
        All = [y for i in self.TopPostFSRChildren for y in self.TopPostFSRChildren[i] if abs(y.pdgid) in [11, 13, 15]]
        for j in self.Leptons:
            dr = 99
            low = ""
            for i in All:
                d = i.DeltaR(j) 
                if dr < d:
                    continue
                dr = d
                low = i
            if low == "":
                continue
            j.Index = low.Index
            self.TopPostFSR[low.Index][0].Decay.append(j)

        self.DetectorParticles = []
        self.DetectorParticles += self.Electrons
        self.DetectorParticles += self.Muons
        self.DetectorParticles += self.Jets

        self.TruthJets = self.DictToList(self.TruthJets)
        self.TruthTops = self.DictToList(self.TruthTops)
        self.TruthTopChildren = self.DictToList(self.TruthTopChildren)

        self.TopPreFSR = self.DictToList(self.TopPreFSR)
        self.TopPostFSR = self.DictToList(self.TopPostFSR)
        self.TopPostFSRChildren = self.DictToList(self.TopPostFSRChildren)

        for i in self.TopPostFSR:
            RecursiveSignal(i.Decay_init, i.FromRes, i.Index)
            RecursiveSignal(i.Decay, i.FromRes, i.Index)

        if ClearVal: 
            del self.Objects
            del self.Leaves
            del self.KeyMap

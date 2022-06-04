from Functions.Particles.Particles import *
from Functions.Event.EventTemplate import EventTemplate

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
        self.mu = "mu"
        self.met = "met_met"
        self.met_phi = "met_phi" 
        self.mu_actual = "mu_actual"
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
                p_i.Index = index
                RecursiveSignal(p_i.Decay_init, sig, index)
                RecursiveSignal(p_i.Decay, sig, index)
       
        for i in self.TruthTops:
            self.TruthTops[i][0].Decay_init += self.TruthTopChildren[i]

        for i in self.TopPostFSR:
            self.TopPostFSR[i][0].Decay_init += self.TopPostFSRChildren[i]

        for i in self.TruthJets:
            self.TruthJets[i][0].GhostTruthJetMap = FixList(self.TruthJets[i][0].GhostTruthJetMap)

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
        
        Leptons = []
        Leptons += self.Electrons
        Leptons += self.Muons
        
        All = [y for i in self.TopPostFSRChildren for y in self.TopPostFSRChildren[i] if abs(y.pdgid) in [11, 13, 15]]
        for j in Leptons:
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

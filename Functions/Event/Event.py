from Functions.Tools.Alerting import Debugging
from Functions.Particles.Particles import *
from Functions.Tools.Variables import VariableManager

class EventVariables:
    def __init__(self):
        self.MinimalTrees = ["tree", "nominal"]
        self.MinimalLeaves = []
        self.MinimalLeaves += Event().Leaves
        self.MinimalLeaves += Event_Custom().Leaves
        self.MinimalLeaves += TruthJet().Leaves
        self.MinimalLeaves += Jet().Leaves
        self.MinimalLeaves += Jet_C().Leaves
        self.MinimalLeaves += Electron().Leaves
        self.MinimalLeaves += Muon().Leaves
        self.MinimalLeaves += Top().Leaves
        self.MinimalLeaves += Truth_Top_Child().Leaves
        self.MinimalLeaves += Truth_Top_Child_Init().Leaves
        self.MinimalLeaves += RCSubJet().Leaves
        self.MinimalLeaves += RCJet().Leaves
        self.MinimalLeaves += TruthJet_C().Leaves
        self.MinimalLeaves += TruthTop_C().Leaves
        self.MinimalLeaves += TruthTopChild_C().Leaves
        self.MinimalLeaves += TopPreFSR_C().Leaves
        self.MinimalLeaves += TopPostFSR_C().Leaves
        self.MinimalLeaves += TopPostFSRChildren_C().Leaves

class Event_Custom(VariableManager):
    def __init__(self):
        VariableManager.__init__(self)
        self.runNumber = "runNumber"
        self.eventNumber = "eventNumber"
        self.mu = "mu"

        self.met = "met_met"
        self.met_phi = "met_phi" 
        self.mu_actual = "mu_actual"
        
        self.Type = "Event"
        self.ListAttributes()
        self.CompileKeyMap()
        self.iter = -1
        self.Tree = ""

        self.Objects = {
                         "Electrons" : Electron(), 
                         "Muons" : Muon(), 
                         "Jets" : Jet_C(), 
                         "TruthJets": TruthJet_C(), 
                         "TruthTops" : TruthTop_C(), 
                         "TruthTopChildren": TruthTopChild_C(), 
                         "TopPreFSR" : TopPreFSR_C(),
                         "TopPostFSR" : TopPostFSR_C(),
                         "TopPostFSRChildren" : TopPostFSRChildren_C()
                        }

        for i in self.Objects:
           self.SetAttribute(i, {})
        self.BrokenEvent = False

    def ParticleProxy(self, File):
        
        def Attributor(variable, value):
            for i in self.Objects:
                obj = self.Objects[i]
                if variable in obj.KeyMap:
                    o = getattr(self, i)
                    o[variable] = value
                    self.SetAttribute(i, o)
                    return True
            return False

        self.ListAttributes()
        for i in File.ArrayLeaves:
            if self.Tree in i:
                var = i.replace(self.Tree + "/", "")
                try: 
                    val = File.ArrayLeaves[i][self.iter]
                except:
                    self.BrokenEvent = True
                    continue
                
                if Attributor(var, val):
                    continue

                if var in Event_Custom().KeyMap:
                    self.SetAttribute(self.KeyMap[var], val)


    def CompileEvent(self, ClearVal = True):
        for i in self.Objects:
            l = getattr(self, i)
            l = CompileParticles(l, self.Objects[i]).Compile(ClearVal)
            self.SetAttribute(i, l)
        
        for i in self.TruthTops:
            self.TruthTops[i][0].Decay_init += self.TruthTopChildren[i]

        for i in self.TopPostFSR:
            self.TopPostFSR[i][0].Decay_init += self.TopPostFSRChildren[i]

        for i in self.TopPostFSR:
            for j in self.TruthJets:

                if isinstance(self.TruthJets[j][0].GhostTruthJetMap, str):
                    self.TruthJets[j][0].GhostTruthJetMap = 0
                
                try:
                    self.TruthJets[j][0].GhostTruthJetMap = [int(self.TruthJets[j][0].GhostTruthJetMap)]
                except TypeError:
                    self.TruthJets[j][0].GhostTruthJetMap = list(self.TruthJets[j][0].GhostTruthJetMap)
               
                if self.TopPostFSR[i][0].Index+1 in self.TruthJets[j][0].GhostTruthJetMap:
                    self.TopPostFSR[i][0].Decay += self.TruthJets[j]
        
        for i in self.Jets:
            l = []
            try:
                l = [int(self.Jets[i][0].JetMapGhost)]
            except:
                if isinstance(self.Jets[i][0].JetMapGhost, str):
                    continue
                else:
                    l = list(self.Jets[i][0].JetMapGhost)
            self.Jets[i][0].JetMapGhost = l 
            for k in l:
                truthj = self.TruthJets[k][0]
                truthj.Decay.append(self.Jets[i][0])
        
        self.Electrons = self.DictToList(self.Electrons)
        self.Muons = self.DictToList(self.Muons)
        self.Jets = self.DictToList(self.Jets)

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

        if ClearVal: 
            del self.Objects
            del self.Leaves
            del self.KeyMap
    
    def DictToList(self, inp): 
        out = []
        for i in inp:
            out += inp[i]
        return out



class Event(VariableManager, Debugging):
    def __init__(self):
        VariableManager.__init__(self)
        Debugging.__init__(self)
        self.runNumber = "runNumber"
        self.eventNumber = "eventNumber"
        self.mu = "mu"
        self.met = "met_met"
        self.phi = "met_phi"
        self.mu_actual = "mu_actual"

        self.Type = "Event"
        self.ListAttributes()
        self.CompileKeyMap()
        self.iter = -1
        self.Tree = ""
        
        self.Objects = {"Electrons" : Electron(), 
                        "Muons" : Muon(), 
                        "TruthJets" : TruthJet(), 
                        "Jets" : Jet(), 
                        "TruthTops" : Top(), 
                        "TruthChildren" : Truth_Top_Child(), 
                        "TruthChildren_init" : Truth_Top_Child_Init(), 
                        "RCSubJets" : RCSubJet(), 
                        "RCJets" : RCJet()}

        self.DetectorParticles = []
        
        for i in self.Objects:
            self.SetAttribute(i, {})

        self.Anomaly = {}
        self.Anomaly_TruthMatch = False
        self.Anomaly_TruthMatch_init = False
        self.Anomaly_Detector = False
        
        self.BrokenEvent = False

    def ParticleProxy(self, File):
        
        def Attributor(variable, value):
            for i in self.Objects:
                obj = self.Objects[i]
                if variable in obj.KeyMap:
                    o = getattr(self, i)
                    o[variable] = value
                    self.SetAttribute(i, o)
                    return True
            return False

        self.ListAttributes()
        for i in File.ArrayLeaves:
            if self.Tree in i:
                var = i.replace(self.Tree + "/", "")
                try: 
                    val = File.ArrayLeaves[i][self.iter]
                except:
                    self.BrokenEvent = True
                
                if Attributor(var, val):
                    continue

                if var in Event().KeyMap:
                    self.SetAttribute(self.KeyMap[var], val)

    def MatchingRule(self, P1, P2, dR):
        
        if P1.Type == "truthjet" and P2.Type == "jet":
            if P1.flavour == P2.truthflav and P1.flavour_extended == P2.truthflavExtended and dR < 0.1:
                return True
        
        # Truth child to truthjet matching conditions
        if "child" in P1.Type and P2.Type == "truthjet":

            # No B or C Hadrons: MC u, d, s are considered 0 flavoured for some reason
            if P2.nCHad == P2.nBHad == 0 and P2.flavour == P2.flavour_extended:
                
                # Coverage for u, d, s quark jets
                if abs(P1.pdgid) < 4 and P2.flavour == 0:
                    return True 
                
                # Coverage for tau lepton truth jets (?) 
                if abs(P1.pdgid) == P2.flavour:
                    return True

            if abs(P1.pdgid) == P2.flavour and abs(P1.pdgid) == P2.flavour_extended and P2.nCHad == P2.nBHad == 1:
                return True

            if abs(P1.pdgid) == P2.flavour and abs(P1.pdgid) == P2.flavour_extended and P2.nCHad == 1 and P2.nBHad == 0:
                return True
            if dR < 0.1:
                return True

        # Truth Child to Detector lepton matching conditions
        if "child" in P1.Type and (P2.Type == "mu" or P2.Type == "el"):
            if P2.true_isPrompt == 1:
                return True


    def DeltaRMatrix(self, List1, List2): 
        delR = {}
        for i in List1:
            for c in List2:
                dR = c.DeltaR(i)
                delR[str(dR) + "_" + str(c.Index) +"_"+str(i.Index)] = [c, i, dR]
        self.dRMatrix = sorted(delR.items())

    def DeltaRLoop(self):
        def AppendToParent(particle, truth):
            if self.__init:
                truth.Decay_init.append(particle)
            else:
                truth.Decay.append(particle)

        captured = []
        All_c = []
        nu = [12, 14, 16]
        for dR in self.dRMatrix:
            j = dR[1][0]
            t = dR[1][1]
            dR = dR[1][2]
                
            if j not in All_c:
                if "child" in j.Type:
                    # veto neutrinos 
                    if abs(j.pdgid) in nu:
                        continue
                All_c.append(j)

            if j in captured:
                continue
            if self.MatchingRule(t, j, dR):
                AppendToParent(j, t)
                captured.append(j)
        
        if len(captured) != len(All_c):
            self.Anomaly[self.CallLoop] = [captured, All_c]
        
            if self.CallLoop == "DetectorMatchingEngine":
                self.Anomaly_Detector = True
            
            if self.CallLoop == "TruthMatchingEngine":
                self.Anomaly_TruthMatch = True
            
            if self.CallLoop == "TruthMatchingEngine_init":
                self.Anomaly_TruthMatch_init = True
    
    def DictToList(self, inp): 
        out = []
        for i in inp:
            out += inp[i]
        return out

    def CompileEvent(self, ClearVal = True):

        self.TruthTops = CompileParticles(self.TruthTops, Top()).Compile(ClearVal) 
        self.TruthChildren_init = CompileParticles(self.TruthChildren_init, Truth_Top_Child_Init()).Compile(ClearVal)
        self.TruthChildren = CompileParticles(self.TruthChildren, Truth_Top_Child()).Compile(ClearVal)
        self.TruthJets = CompileParticles(self.TruthJets, TruthJet()).Compile(ClearVal)
        self.Jets = CompileParticles(self.Jets, Jet()).Compile(ClearVal)
        self.Muons = CompileParticles(self.Muons, Muon()).Compile(ClearVal)
        self.Electrons = CompileParticles(self.Electrons, Electron()).Compile(ClearVal)
        self.RCSubJets = CompileParticles(self.RCSubJets, RCSubJet()).Compile(ClearVal)
        self.RCJets = CompileParticles(self.RCJets, RCJet()).Compile(ClearVal)

        for i in self.TruthTops:
            self.TruthTops[i][0].Decay_init = self.TruthChildren_init[i]
            self.TruthTops[i][0].Decay = self.TruthChildren[i]

        for i in self.RCJets:
            self.RCJets[i][0].Constituents = self.RCSubJets[i]


        # Very important to have this one first since the 
        # truth partons contains information about the pdgid (truth partons)
        if self.Debug:
            print(">>>>>>>========== New Event ==========")

        self.__init = False
        self.DetectorMatchingEngine()
        
        self.__init = True
        self.TruthMatchingEngine()

        self.__init = False
        self.TruthMatchingEngine()

        self.RCSubJetMatchingEngine()
        
        self.TruthTops = self.DictToList(self.TruthTops)
        self.TruthChildren = self.DictToList(self.TruthChildren)
        self.TruthChildren_init = self.DictToList(self.TruthChildren_init)

        self.TruthJets = self.DictToList(self.TruthJets)
        self.Jets = self.DictToList(self.Jets)
        self.Muons = self.DictToList(self.Muons)
        self.Electrons = self.DictToList(self.Electrons)

        self.RCJets = self.DictToList(self.RCJets)
        self.RCSubJets = self.DictToList(self.RCSubJets)

        self.DetectorParticles += self.Jets
        self.DetectorParticles += self.Muons
        self.DetectorParticles += self.Electrons
        self.DetectorParticles += self.RCJets
        self.DetectorParticles += self.RCSubJets
        
        for i in self.TruthTops:
            i.PropagateSignalLabel()

        for i in self.RCJets:
            i.PropagateJetSignal()
        
        for i in self.Jets:
            for j in i.Decay:
                j.Flav = i.truthflavExtended
        
        if ClearVal: 
            del self.dRMatrix
            del self.Objects
            del self.Leaves
            del self.KeyMap

    def DetectorMatchingEngine(self):
        DetectorParticles = []
        DetectorParticles += self.DictToList(self.Jets)
        DetectorParticles += self.DictToList(self.Electrons)
        DetectorParticles += self.DictToList(self.Muons)
        TruthJets = self.DictToList(self.TruthJets)
        self.CallLoop = "DetectorMatchingEngine"
        
        self.DeltaRMatrix(TruthJets, DetectorParticles)
        if self.Debug:
            DetectorParticles += TruthJets
            self.DebugTruthDetectorMatch(DetectorParticles)
        else:
            self.DeltaRLoop()
    
    def RCSubJetMatchingEngine(self):
        
        Jets = self.DictToList(self.Jets)
        RC = self.DictToList(self.RCSubJets)
        self.DeltaRMatrix(Jets, RC)
        
        captured = []
        for i in self.dRMatrix:
            if i[1][0] not in captured:
                i[1][1].Decay.append(i[1][0])
                captured.append(i[1][0])

    def TruthMatchingEngine(self):
        Truth = []
        if self.__init:
            Truth += self.DictToList(self.TruthChildren_init)
            self.CallLoop = "TruthMatchingEngine_init"       
        else:
            Truth += self.DictToList(self.TruthChildren)
            self.CallLoop = "TruthMatchingEngine"       
       
        All = []
        All += self.DictToList(self.TruthJets)
        All += self.DictToList(self.Electrons)
        All += self.DictToList(self.Muons)
        
        self.DeltaRMatrix(Truth, All)       
        if self.Debug:
            All += Truth
            self.DebugTruthDetectorMatch(All)
        else:
            self.DeltaRLoop() 
        
   
        

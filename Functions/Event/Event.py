from Functions.IO.IO import UpROOT_Reader
from Functions.Tools.Alerting import Debugging
from Functions.Particles.Particles import *
from Functions.Tools.Variables import VariableManager
from Functions.Tools.DataTypes import DataTypeCheck

class EventVariables:
    def __init__(self):
        self.MinimalTree = ["nominal"] #, "MUON_SCALE__1up"]
        self.MinimalBranch = []
        self.MinimalBranch += Event().Branches
        self.MinimalBranch += TruthJet().Branches
        self.MinimalBranch += Jet().Branches
        self.MinimalBranch += Electron().Branches
        self.MinimalBranch += Muon().Branches
        self.MinimalBranch += Top().Branches
        self.MinimalBranch += Truth_Top_Child().Branches
        self.MinimalBranch += Truth_Top_Child_Init().Branches

class Event(VariableManager, DataTypeCheck, Debugging):
    def __init__(self):
        VariableManager.__init__(self)
        DataTypeCheck.__init__(self)
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
        
        self.TruthTops = {}
        self.TruthChildren_init = {}
        self.TruthChildren = {}
        self.TruthJets = {}
        self.Jets = {}
        self.Muons = {}
        self.Electrons = {}

        self.TruthAssignedToChild_init = []
        self.TruthAssignedToChild = []
        self.TruthNotAssigned = []

        self.Release = False
    
    def ParticleProxy(self, File):
        self.ListAttributes()
        el = Electron()
        mu = Muon()
        truthjet = TruthJet()
        jet = Jet()
        top = Top()
        child = Truth_Top_Child()
        child_init = Truth_Top_Child_Init()
        event = Event()

        for i in File.ArrayBranches:
            if self.Tree in i:
                var = i.replace(self.Tree + "/", "")
                val = File.ArrayBranches[i][self.iter]
                if var in truthjet.KeyMap:
                    self.TruthJets[var] = val 
                elif var in jet.KeyMap:
                    self.Jets[var] = val
                elif var in el.KeyMap:
                    self.Electrons[var] = val
                elif var in mu.KeyMap:
                    self.Muons[var] = val
                elif var in child.KeyMap:
                    self.TruthChildren[var] = val
                elif var in child_init.KeyMap:
                    self.TruthChildren_init[var] = val
                elif var in top.KeyMap:
                    self.TruthTops[var] = val
                elif var in event.KeyMap:
                    self.SetAttribute(self.KeyMap[var], val)


    def MatchingRule(self, P1, P2, dR):
        
        if P1.Type == "truthjet" and P2.Type == "jet":

            if P1.flavour == P2.truthflav and P1.flavour_extended == P2.truthflavExtended:
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

        # Truth Child to Detector lepton matching conditions
        if "child" in P1.Type and (P2.Type == "mu" or P2.Type == "el"):
            if P2.true_isPrompt == 1:
                return True


    def DeltaRMatrix(self, List1, List2): 
        delR = {}
        for i in List1:
            for c in List2:
                delR[c.DeltaR(i)] = [c, i]
        self.dRMatrix = sorted(delR.items())



    def CompileEvent(self):
        self.TruthTops = CompileParticles(self.TruthTops, Top()).Compile() 
        self.TruthChildren_init = CompileParticles(self.TruthChildren_init, Truth_Top_Child_Init()).Compile()
        self.TruthChildren = CompileParticles(self.TruthChildren, Truth_Top_Child()).Compile()
        self.TruthJets = CompileParticles(self.TruthJets, TruthJet()).Compile()
        self.Jets = CompileParticles(self.Jets, Jet()).Compile()
        self.Muons = CompileParticles(self.Muons, Muon()).Compile()
        self.Electrons = CompileParticles(self.Electrons, Electron()).Compile()

        for i in self.TruthTops:
            self.TruthTops[i][0].Decay_init = self.TruthChildren_init[i]
            self.TruthTops[i][0].Decay = self.TruthChildren[i]

        All = []
        All += self.DictToList(self.Muons)
        All += self.DictToList(self.Electrons)
        All += self.DictToList(self.TruthJets)
        self.__All = All
        
        Truth_init = self.DictToList(self.TruthChildren_init)
        Truth = self.DictToList(self.TruthChildren) 

        self.DeltaRMatrix(Truth_init, All)
        self.__init = True
        self.TruthMatchingEngine()

        self.DeltaRMatrix(All, Truth)
        self.__init = False
        self.TruthMatchingEngine()

        self.DetectorMatchingEngine()

        if self.Release:
            self.TruthTops = self.DictToList(self.TruthTops)
            self.TruthJets = self.DictToList(self.TruthJets)
            self.Jets = self.DictToList(self.Jets)
            self.Muons = self.DictToList(self.Muons)
            self.Electrons = self.DictToList(self.Electrons)

    def TruthMatchingEngine(self):
        def AppendToParent(particle, truth):
            if self.__init:
                truth.Decay_init.append(particle)
                self.TruthAssignedToChild_init.append(particle)
            else:
                truth.Decay.append(particle)
                self.TruthAssignedToChild.append(particle)
            particle.ParentPDGID = tc.pdgid       
        
        AllParticles = []
        AllParticles += self.__All
        AllParticles += self.DictToList(self.TruthChildren_init)
        
        #self.DebugTruthDetectorMatch(AllParticles) 
        captured = []
        for dR in self.dRMatrix:
            tjet = dR[1][0]
            tc = dR[1][1]
            dR = dR[0]
 
            if self.MatchingRule(tc, tjet, dR):
                AppendToParent(tjet, tc)
                captured.append(tjet) 
            
            if tjet in captured:
                continue

    def DetectorMatchingEngine(self):
        DetectorParticles = []
        DetectorParticles += self.DictToList(self.Jets)
        DetectorParticles += self.DictToList(self.Electrons)
        DetectorParticles += self.DictToList(self.Muons)
        TruthJets = self.DictToList(self.TruthJets)
        self.DeltaRMatrix(TruthJets, DetectorParticles)

        DetectorParticles += TruthJets
        self.DebugTruthDetectorMatch(DetectorParticles)

    


   
class EventGenerator(UpROOT_Reader, Debugging, EventVariables):
    def __init__(self, dir, Verbose = True, DebugThresh = -1):
        UpROOT_Reader.__init__(self, dir, Verbose)
        Debugging.__init__(self, Threshold = DebugThresh)
        EventVariables.__init__(self)
        self.Events = []
        
    def SpawnEvents(self, Full = False):
        self.Read()
        

        for f in self.FileObjects:
            Tree = []
            Branches = []
            self.Notify("Reading -> " + f)
            F = self.FileObjects[f]
            
            if Full:
                F.ScanFull()
                Trees = F.AllTrees
                Branches = F.AllBranches
            else:
                Trees = self.MinimalTree
                Branches = self.MinimalBranch
            F.Trees = Trees
            F.Branches = Branches
            F.CheckKeys()
            F.ConvertToArray()
                

            ## =====!!!! TEMP!!!! DELETE!!!!! ======#
            #import pickle
            #outfile = open("./File", "wb")
            #pickle.dump(F, outfile)
            #outfile.close()

            #infile = open("./File", "rb")
            #F = pickle.load(infile)
            #infile.close()

            
            for i in range(len(F.ArrayBranches["nominal/eventNumber"])):
                pairs = {}
                for k in Trees:
                    E = Event()
                    E.Tree = k
                    E.iter = i
                    E.ParticleProxy(F)
                    E.CompileEvent()
                    pairs[k] = E
                
                self.Events.append(pairs) 
                if self.TimeOut():
                    break 
            
            break

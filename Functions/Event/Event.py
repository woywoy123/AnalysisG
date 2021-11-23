from Functions.IO.IO import UpROOT_Reader
from Functions.Tools.Alerting import Debugging
from Functions.Particles.Particles import *
from Functions.Tools.Variables import VariableManager
from Functions.Tools.DataTypes import DataTypeCheck, TemplateThreading, Threading
import math

class EventVariables:
    def __init__(self):
        self.MinimalTree = ["tree", "nominal"]
        self.MinimalBranch = []
        self.MinimalBranch += Event().Branches
        self.MinimalBranch += TruthJet().Branches
        self.MinimalBranch += Jet().Branches
        self.MinimalBranch += Electron().Branches
        self.MinimalBranch += Muon().Branches
        self.MinimalBranch += Top().Branches
        self.MinimalBranch += Truth_Top_Child().Branches
        self.MinimalBranch += Truth_Top_Child_Init().Branches
        self.MinimalBranch += RCSubJet().Branches
        self.MinimalBranch += RCJet().Branches

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
        
        self.Objects = {"el->Electrons" : Electron(), "mu->Muons" : Muon(), "truthjet->TruthJets" : TruthJet(), 
                       "jet->Jets" : Jet(), "top->TruthTops" : Top(), "child->TruthChildren" : Truth_Top_Child(), 
                       "child_init->TruthChildren_init" : Truth_Top_Child_Init(), "rcsubjet->RCSubJets" : RCSubJet(), 
                       "rcjet->RCJets" : RCJet()}
        for i in self.Objects:
            self.SetAttribute(i.split("->")[1], {})

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
                    o = getattr(self, i.split("->")[1])
                    o[variable] = value
                    self.SetAttribute(i.split("->")[1], o)
                    return True
            return False

        self.ListAttributes()
        for i in File.ArrayBranches:
            if self.Tree in i:
                var = i.replace(self.Tree + "/", "")
                try: 
                    val = File.ArrayBranches[i][self.iter]
                except:
                    self.BrokenEvent = True
                    continue
                
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
    
    def CompileSpecificParticles(self, particles = False):
        if particles == "TruthTops":
            self.TruthTops = self.DictToList(CompileParticles(self.TruthTops, Top()).Compile())

        if particles == "TruthChildren":
            self.TruthTops = CompileParticles(self.TruthTops, Top()).Compile() 
            self.TruthChildren = CompileParticles(self.TruthChildren, Truth_Top_Child()).Compile()
            self.TruthChildren_init = CompileParticles(self.TruthChildren_init, Truth_Top_Child_Init()).Compile()
        
            for i in self.TruthTops:
                self.TruthTops[i][0].Decay_init = self.TruthChildren_init[i]
                self.TruthTops[i][0].Decay = self.TruthChildren[i]
            
            self.TruthChildren = self.DictToList(self.TruthChildren)
            self.TruthChildren_init = self.DictToList(self.TruthChildren_init)
            self.TruthTops = self.DictToList(self.TruthTops)
            for i in self.TruthTops:
                i.PropagateSignalLabel()

        if particles == "TruthJets":
            self.TruthJets = self.DictToList(CompileParticles(self.TruthJets, TruthJet()).Compile())
        
        if particles == "Detector":
            self.Jets = self.DictToList(CompileParticles(self.Jets, Jet()).Compile())
            self.Muons = self.DictToList(CompileParticles(self.Muons, Muon()).Compile())
            self.Electrons = self.DictToList(CompileParticles(self.Electrons, Electron()).Compile())
            self.RCSubJets = self.DictToList(CompileParticles(self.RCSubJets, RCSubJet()).Compile())
            self.RCJets = self.DictToList(CompileParticles(self.RCJets, RCJet()).Compile())

    def CompileEvent(self):
        self.TruthTops = CompileParticles(self.TruthTops, Top()).Compile() 
        self.TruthChildren_init = CompileParticles(self.TruthChildren_init, Truth_Top_Child_Init()).Compile()
        self.TruthChildren = CompileParticles(self.TruthChildren, Truth_Top_Child()).Compile()
        self.TruthJets = CompileParticles(self.TruthJets, TruthJet()).Compile()
        self.Jets = CompileParticles(self.Jets, Jet()).Compile()
        self.Muons = CompileParticles(self.Muons, Muon()).Compile()
        self.Electrons = CompileParticles(self.Electrons, Electron()).Compile()
        self.RCSubJets = CompileParticles(self.RCSubJets, RCSubJet()).Compile()
        self.RCJets = CompileParticles(self.RCJets, RCJet()).Compile()

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

        self.DetectorParticles = []
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
         
        del self.dRMatrix
        del self.Objects
        del self.Branches
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
        
   
class EventGenerator(UpROOT_Reader, Debugging, EventVariables):
    def __init__(self, dir, Verbose = True, DebugThresh = -1, Debug = False):
        UpROOT_Reader.__init__(self, dir, Verbose)
        Debugging.__init__(self, Threshold = DebugThresh)
        EventVariables.__init__(self)
        self.Events = {}
        self.FileEventIndex = {}
        self.__Debug = Debug
        self.Threads = 12
        
    def SpawnEvents(self):

        self.Read()
        ind = 0
        for f in self.FileObjects:
            
            self.Notify("_______NEW FILE______")
            self.Notify("Reading -> " + f)
            F = self.FileObjects[f]
            
            F.Trees = self.MinimalTree
            F.Branches = self.MinimalBranch
            F.CheckKeys()
            F.ConvertToArray()
            FirstBranches = list(F.ObjectBranches)[0]

            Trees = []
            for k in F.ObjectBranches:
                if k.split("/")[0] not in Trees:
                    Trees.append(k.split("/")[0])

            self.Notify("SPAWNING EVENTS IN FILE -> " + f)
            self.Events[f] = []
            end = False
            for i in range(len(F.ArrayBranches[FirstBranches])):
                pairs = {}
                for k in Trees:
                    if self.TimeOut():
                        end = True
                        break

                    E = Event()
                    if self.__Debug == True:
                        E.Debug = True
        
                    E.Tree = k
                    E.iter = i
                    E.ParticleProxy(F)
                    pairs[k] = E
                if end: 
                    break
                self.Events[f].append(pairs)

            del F
        self.FileObjects = {}

    def CompileEvent(self, SingleThread = False, particle = False):
        
        def function(Entries):
            for k in Entries:
                for j in k:
                    k[j].CompileEvent()
            return Entries

        def Loop(Entries):
            for k in Entries:
                for j in k:
                    k[j].CompileSpecificParticles(particle_)
            return Entries
  
        self.Caller = "EVENTCOMPILER"
        
        it = 0
        ev = {}
        for f in self.Events:
            self.Notify("COMPILING EVENTS IN FILE -> " + f)
            
            Events = self.Events[f]
            entries_percpu = math.ceil(len(Events) / self.Threads)

            self.Batches = {}
            Thread = []
            for k in range(self.Threads):
                self.Batches[k] = [] 
                for i in Events[k*entries_percpu : (k+1)*entries_percpu]:
                    self.Batches[k].append(i)

                if particle == False:
                    Thread.append(TemplateThreading(k, "", "Batches", self.Batches[k], function))
                else: 
                    particle_ = particle
                    Thread.append(TemplateThreading(k, "", "Batches", self.Batches[k], Loop))
            th = Threading(Thread, self.Threads)
            th.Verbose = True
            if SingleThread:
                th.TestWorker()
            else:
                th.StartWorkers()

            self.Notify("FINISHED COMPILING EVENTS IN FILE -> " + f)
            self.Notify("SORTING INTO DICTIONARY -> " + f)
            res = th.Result
            for i in res:
                i.SetAttribute(self)
            
            self.FileEventIndex[f] = []
            self.FileEventIndex[f].append(it)
            for k in self.Batches:
                for j in self.Batches[k]:
                    ev[it] = j
                    it += 1
            self.FileEventIndex[f].append(it-1)
            
            del th
            self.Batches = {}
            Thread = []
        del self.MinimalTree 
        self.Events = ev

    def EventIndexFileLookup(self, index):

        for i in self.FileEventIndex:
            min_ = self.FileEventIndex[i][0]
            max_ = self.FileEventIndex[i][1]

            if index >= min_ and index <= max_:
                return i

        

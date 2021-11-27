from Functions.IO.IO import PickleObject, UnpickleObject, File, Directories
from Functions.Tools.Alerting import Debugging
from Functions.Particles.Particles import *
from Functions.Tools.Variables import VariableManager
from Functions.Tools.DataTypes import TemplateThreading, Threading
import math

class EventVariables:
    def __init__(self):
        self.MinimalTrees = ["tree", "nominal"]
        self.MinimalLeaves = []
        self.MinimalLeaves += Event().Leaves
        self.MinimalLeaves += TruthJet().Leaves
        self.MinimalLeaves += Jet().Leaves
        self.MinimalLeaves += Electron().Leaves
        self.MinimalLeaves += Muon().Leaves
        self.MinimalLeaves += Top().Leaves
        self.MinimalLeaves += Truth_Top_Child().Leaves
        self.MinimalLeaves += Truth_Top_Child_Init().Leaves
        self.MinimalLeaves += RCSubJet().Leaves
        self.MinimalLeaves += RCJet().Leaves

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
        
        self.Objects = {"el->Electrons" : Electron(), "mu->Muons" : Muon(), "truthjet->TruthJets" : TruthJet(), 
                       "jet->Jets" : Jet(), "top->TruthTops" : Top(), "child->TruthChildren" : Truth_Top_Child(), 
                       "child_init->TruthChildren_init" : Truth_Top_Child_Init(), "rcsubjet->RCSubJets" : RCSubJet(), 
                       "rcjet->RCJets" : RCJet()}

        self.DetectorParticles = []
        
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
        
   
class EventGenerator(Debugging, EventVariables, Directories):
    def __init__(self, dir, Verbose = True, Start = 0, Stop = -1, Debug = False):
        Debugging.__init__(self, Threshold = Stop - Start)
        EventVariables.__init__(self)
        Directories.__init__(self, dir)
        self.Events = {}
        self.FileEventIndex = {}
        self.__Debug = Debug
        self.Threads = 12
        self.Caller = "EVENTGENERATOR"
        self.__Start = Start
        self.__Stop = Stop
        
    def SpawnEvents(self):
        self.GetFilesInDir()
        for i in self.Files:
            self.Notify("_______NEW DIRECTORY______: " + str(i))
            for F in self.Files[i]:
                self.Events[i + "/" + F] = []
                F_i = File(i + "/" + F, self.__Debug)
                F_i.Trees = self.MinimalTrees
                F_i.Leaves = self.MinimalLeaves
                F_i.CheckKeys()
                F_i.ConvertToArray()
                
                self.Notify("SPAWNING EVENTS IN FILE -> " + F)
                for l in range(len(F_i.ArrayLeaves[list(F_i.ArrayLeaves)[0]])):
                    pairs = {}
                    for tr in F_i.ObjectTrees:
                        
                        if self.__Start != 0:
                            if self.__Start <= l:
                                self.Count() 
                            else:
                                continue
                        else: 
                            self.Count()

                        E = Event()
                        E.Debug = self.__Debug
                        E.Tree = tr
                        E.iter = l
                        E.ParticleProxy(F_i)
                        pairs[tr] = E

                    self.Events[i + "/" + F].append(pairs)
                    
                    if self.Stop():
                        self.ResetCounter()
                        break
                del F_i
                
        del self.MinimalLeaves
        del self.MinimalTrees

    def CompileEvent(self, SingleThread = False, ClearVal = True):
        
        def function(Entries):
            for k in Entries:
                for j in k:
                    k[j].CompileEvent(ClearVal = ClearVal)
            return Entries

        self.Caller = "EVENTCOMPILER"
        
        it = 0
        ev = {}
        for f in self.Events:
            self.Notify("COMPILING EVENTS FROM FILE -> " + f)
            
            Events = self.Events[f]
            entries_percpu = math.ceil(len(Events) / self.Threads)

            self.Batches = {}
            Thread = []
            for k in range(self.Threads):
                self.Batches[k] = [] 
                for i in Events[k*entries_percpu : (k+1)*entries_percpu]:
                    self.Batches[k].append(i)

                Thread.append(TemplateThreading(k, "", "Batches", self.Batches[k], function))
            th = Threading(Thread, self.Threads)
            th.Verbose = True
            if SingleThread:
                th.TestWorker()
            else:
                th.StartWorkers()

            self.Notify("FINISHED COMPILING EVENTS FROM FILE -> " + f)
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
        self.Events = ev

    def EventIndexFileLookup(self, index):

        for i in self.FileEventIndex:
            min_ = self.FileEventIndex[i][0]
            max_ = self.FileEventIndex[i][1]

            if index >= min_ and index <= max_:
                return i

        

from Functions.IO.IO import UpROOT_Reader
from Functions.Tools.Alerting import Debugging
from Functions.Particles.Particles import *
from Functions.Tools.Variables import VariableManager
from Functions.Tools.DataTypes import DataTypeCheck, TemplateThreading, Threading
import math

class EventVariables:
    def __init__(self):
        self.MinimalTree = ["nominal"]
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
    def __init__(self, Debug = False):
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
        self.Anomaly = {}
        self.Anomaly_TruthMatch = False
        self.Anomaly_TruthMatch_init = False
        self.Anomaly_Detector = False
        
        self.BrokenEvent = False
    
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
                try: 
                    val = File.ArrayBranches[i][self.iter]
                except:
                    self.BrokenEvent = True
                    continue

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
            self.TruthTops = CompileParticles(self.TruthTops, Top()).Compile() 

        if particles == "TruthChildren":
            self.TruthChildren_init = CompileParticles(self.TruthChildren_init, Truth_Top_Child_Init()).Compile()
            self.TruthChildren = CompileParticles(self.TruthChildren, Truth_Top_Child()).Compile()

        if particles == "TruthJets":
            self.TruthJets = CompileParticles(self.TruthJets, TruthJet()).Compile()
        
        if particles == "Detector":
            self.Jets = CompileParticles(self.Jets, Jet()).Compile()
            self.Muons = CompileParticles(self.Muons, Muon()).Compile()
            self.Electrons = CompileParticles(self.Electrons, Electron()).Compile()

        self.TruthTops = self.DictToList(self.TruthTops)
        self.TruthJets = self.DictToList(self.TruthJets)
        self.Jets = self.DictToList(self.Jets)
        self.Muons = self.DictToList(self.Muons)
        self.Electrons = self.DictToList(self.Electrons)






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


        self.TruthTops = self.DictToList(self.TruthTops)
        self.TruthJets = self.DictToList(self.TruthJets)
        self.Jets = self.DictToList(self.Jets)
        self.Muons = self.DictToList(self.Muons)
        self.Electrons = self.DictToList(self.Electrons)


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
        self.__Debug = Debug
        
    def SpawnEvents(self, Full = False):

        self.Read()
        for f in self.FileObjects:

            Trees = []
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

            self.Events[F.FileName] = []
            
            self.Notify("SPAWNING EVENTS IN FILE -> " + f)
            for i in range(len(F.ArrayBranches["nominal/eventNumber"])):
                pairs = {}
                for k in Trees:
                    E = Event()
                    if self.__Debug == True:
                        E.Debug = True

                    E.Tree = k
                    E.iter = i
                    E.ParticleProxy(F)
                    pairs[k] = E
                
                self.Events[F.FileName].append(pairs)
                
                if self.TimeOut():
                    break 
            del F
        self.FileObjects = {}

    def CompileEvent(self, SingleThread = False, particle = ""):
        
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

        threads = 12
        for f in self.Events:
            self.Notify("COMPILING EVENTS IN FILE -> " + f)
            
            Events = self.Events[f]
            entries_percpu = math.ceil(len(Events) / threads)

            self.Batches = {}
            Thread = []
            for k in range(threads):
                self.Batches[k] = [] 
                for i in Events[k*entries_percpu : (k+1)*entries_percpu]:
                    self.Batches[k].append(i)

                if particle == False:
                    Thread.append(TemplateThreading(k, "", "Batches", self.Batches[k], function))
                else: 
                    particle_ = particle
                    Thread.append(TemplateThreading(k, "", "Batches", self.Batches[k], Loop))

            th = Threading(Thread, threads)
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
            
            self.Events[f] = {}
            it = 0
            for k in self.Batches:
                for j in self.Batches[k]:
                    self.Events[f][it] = j
                    it += 1
            
            del th
            self.Batches = {}
            Thread = []

        if len(self.Events) == 1:
            for i in self.Events:
                self.Events = self.Events[i]
                break

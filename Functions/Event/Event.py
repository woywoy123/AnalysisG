from Functions.IO.IO import UpROOT_Reader
from Functions.Tools.Alerting import Debugging
from Functions.Particles.Particles import *
from Functions.Tools.Variables import VariableManager

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

class Event(VariableManager):
    def __init__(self):
        VariableManager.__init__(self)
        self.runNumber = "runNumber"
        self.eventNumber = "eventNumber"
        self.mu = "mu"
        self.mu_actual = "mu_actual"

        self.Type = "Event"
        self.ListAttributes()
        self.iter = -1
        self.Tree = ""
        
        self.__TruthTops = {}
        self.__TruthChildren_init = {}
        self.__TruthChildren = {}
        self.__TruthJets = {}
        self.__Jets = {}
        self.__Muons = {}
        self.__Electrons = {}
    
    def ParticleProxy(self, File):
        self.ListAttributes()
        for i in File.ArrayBranches:
            if self.Tree in i:
                var = i.replace(self.Tree + "/", "")
                val = File.ArrayBranches[i][self.iter]
                if "truthjet_" in var:
                    self.__TruthJets[var] = val 
                elif "jet_" in var:
                    self.__Jets[var] = val
                elif "el_" in var:
                    self.__Electrons[var] = val
                elif "mu_" in var:
                    self.__Muons[var] = val
                elif "top_child_" in var:
                    self.__TruthChildren[var] = val
                elif "top_initialState_child" in var:
                    self.__TruthChildren_init[var] = val
                elif "top_" in var:
                    self.__TruthTops[var] = val
    


    def CompileEvent(self):
        self.__TruthTops = CompileParticles(self.__TruthTops, Top()).Compile() 
        self.__TruthChildren_init = CompileParticles(self.__TruthChildren_init, Truth_Top_Child_Init()).Compile()
        self.__TruthChildren = CompileParticles(self.__TruthChildren, Truth_Top_Child()).Compile()
        self.__TruthJets = CompileParticles(self.__TruthJets, TruthJet()).Compile()
        self.__Jets = CompileParticles(self.__Jets, Jet()).Compile()
        self.__Muons = CompileParticles(self.__Muons, Muon()).Compile()
        self.__Electrons = CompileParticles(self.__Electrons, Electron()).Compile()

        for i in self.__TruthTops:
            self.__TruthTops[i][0].DecayParticles["init_child"] = self.__TruthChildren_init[i]
            self.__TruthTops[i][0].DecayParticles["child"] = self.__TruthChildren[i]
        
        All = [] 
        for i in self.__Muons:
            All += self.__Muons[i]
        
        for i in self.__Jets:
            All += self.__Jets[i]

        for i in self.__Electrons:
            All += self.__Electrons[i]

        for i in All:
            print(i.Type, i.pt, i.phi, i.eta, i.e)







class EventGenerator(UpROOT_Reader, Debugging, EventVariables):
    def __init__(self, dir, Verbose = True, DebugThresh = -1):
        UpROOT_Reader.__init__(self, dir, Verbose)
        Debugging.__init__(self, Threshold = DebugThresh)
        EventVariables.__init__(self)
        self.Events = []
        
    def SpawnEvents(self, Full = False):
        self.GetFilesInDir()
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
            
            for i in range(len(F.ArrayBranches["nominal/eventNumber"])):
                pairs = {}
                for k in Trees:
                    E = Event()
                    E.Tree = k
                    E.iter = i
                    E.ParticleProxy(F)
                    E.CompileEvent()
                    pairs[k] = E
                break 
                self.Events.append(pairs) 
            break

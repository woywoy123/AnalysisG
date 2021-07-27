from Functions.IO.IO import UpROOT_Reader
from Functions.Tools.Alerting import Debugging
from Functions.Particles.Particles import *

class EventVariables:
    def __init__(self):
        self.MinimalTree = ["nominal"]
        self.MinimalBranch = ["eventNumber", "runNumber"]
        self.MinimalBranch += TruthJet().Branches
        self.MinimalBranch += Jet().Branches
        self.MinimalBranch += Electron().Branches
        self.MinimalBranch += Muon().Branches
        self.MinimalBranch += Top().Branches
        self.MinimalBranch += Truth_Top_Child().Branches
        self.MinimalBranch += Truth_Top_Child_Init().Branches

class Event:
    def __init__(self):
        self.runNumber = ""
        self.eventNumber = ""
        self.Tree = ""

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
            
            print(F.ArrayBranches)
            for i in range(len(F.ArrayBranches["nominal/eventNumber"])):
                print(i)
                
            



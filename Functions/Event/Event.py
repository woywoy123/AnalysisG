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

class Event(VariableManager, DataTypeCheck):
    def __init__(self):
        VariableManager.__init__(self)
        DataTypeCheck.__init__(self)
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
        
        Truth_init = self.DictToList(self.TruthChildren_init)
        Truth = self.DictToList(self.TruthChildren) 

        self.DeltaRMatrix(All, Truth)
        self.__All = All
        self.__init = True
        self.MatchingEngine()


        if self.Release:
            self.TruthTops = self.DictToList(self.TruthTops)
            self.TruthJets = self.DictToList(self.TruthJets)
            self.Jets = self.DictToList(self.Jets)
            self.Muons = self.DictToList(self.Muons)
            self.Electrons = self.DictToList(self.Electrons)

    def MatchingEngine(self):
        
        # Remove Neutrinos 
        veto_Nu = [12, 14, 16]
        
        captured = []
        remainder = []
        print(">>>>>>>========== New Event ==========")
        print("    --- All Particles ---  ")
        for i in self.__All:
            i.CalculateMass()
            if i.Type == "truthjet":
                print(i.flavour, "--", i.flavour_extended, "---", i.Mass)
            else:
                print(i.Type, "---", i.Mass)




        for dR in self.__Matrix:
            tc = dR[1][0]
            al = dR[1][1]
            dR = dR[0]
            
            if abs(tc.pdgid) in veto_Nu:
                continue
            if al in captured or dR > 0.1:
                continue

            if al.Type == "truthjet":
                if al.flavour == al.flavour_extended and al.flavour == abs(tc.pdgid):
                    print(tc.pdgid, "-> ", al.flavour, round(dR, 4))
                    if self.__init:
                        tc.Decay_init.append(al)
                    else:
                        tc.Decay.append(al)
                    captured.append(al)
                
                elif al.flavour == al.flavour_extended and al.flavour == 0 and abs(tc.pdgid) < 4:
                    print(tc.pdgid, " +---> ", al.flavour, round(dR, 4))
                    captured.append(al)

                #elif al.flavour != al.flavour_extended:
                #     print(tc.pdgid, "!!-> ", al.flavour, " ", al.flavour_extended, " ",round(dR, 4))

                   
            
            if al.Type == "mu" or al.Type == "el":
                if al.true_isPrompt != 1:
                    continue
                print(">>>>>", tc.pdgid, "-> ", al.Type, round(dR, 4))
                if self.__init:
                    tc.Decay_init.append(al)
                else:
                    tc.Decay.append(al)
                captured.append(al)               





    def DeltaRMatrix(self, List1, List2): 
        delR = {}
        for i in List1:
            for c in List2:
                delR[c.DeltaR(i)] = [c, i]
        self.__Matrix = sorted(delR.items())

   
class EventGenerator(UpROOT_Reader, Debugging, EventVariables):
    def __init__(self, dir, Verbose = True, DebugThresh = -1):
        UpROOT_Reader.__init__(self, dir, Verbose)
        Debugging.__init__(self, Threshold = DebugThresh)
        EventVariables.__init__(self)
        self.Events = []
        
    def SpawnEvents(self, Full = False):
        self.Read()
        

        for f in self.FileObjects:
            Trees = ["nominal"] # Tree = []
            Branches = []
            self.Notify("Reading -> " + f)
            #F = self.FileObjects[f]
            #
            #if Full:
            #    F.ScanFull()
            #    Trees = F.AllTrees
            #    Branches = F.AllBranches
            #else:
            #    Trees = self.MinimalTree
            #    Branches = self.MinimalBranch
            #F.Trees = Trees
            #F.Branches = Branches
            #F.CheckKeys()
            #F.ConvertToArray()
                

            ## =====!!!! TEMP!!!! DELETE!!!!! ======#
            import pickle
            #outfile = open("./File", "wb")
            #pickle.dump(F, outfile)
            #outfile.close()

            infile = open("./File", "rb")
            F = pickle.load(infile)
            infile.close()

            
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

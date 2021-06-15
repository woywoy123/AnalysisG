from BaseFunctions.VariableManager import *
from BaseFunctions.Alerting import ErrorAlert, Alerting, Debugging
from BaseFunctions.ParticlesManager import * 
import BaseFunctions
import multiprocessing
from time import sleep

class Event:
    def __init__(self):
        self.TruthMatch = []
        self.DetectorParticles = []
        self.MET = ""
        self.MET_Phi = ""
        self.EventNumber = ""

class EventCompiler(ErrorAlert, BranchVariable, Debugging):
    def __init__(self, FileDir, Tree, Branches, Verbose = True, Debug = False):
        
        ErrorAlert.__init__(self)
        self.expected = str
        self.given = Tree
        self.WrongInputType("GAVE LIST EXPECTED STRING FOR TREE!")

        BranchVariable.__init__(self, FileDir, Tree, Branches)
        self.EventDictionary = {}
        self.__Verbose = Verbose
        Debugging.__init__(self, events = 100, debug = Debug)

    def GenerateEvents(self):
        al = Alerting(self.EventNumber, self.__Verbose)

        al.Notice("ENTERING EVENT GENERATOR")
        for i in range(len(self.EventNumber)):
                
            P = CreateParticleObjects(self.E[i], self.Pt[i], self.Phi[i], self.Eta[i], self.Type)
            VariableObjectProxy(self, P, i) 

            self.EventDictionary[self.EventNumber[i]] = P
            al.ProgressAlert()
            
            self.BreakCounter()
            if self.DebugKill:
                break

        al.Notice("FINISHED EVENT GENERATOR")
        self.MultiThreadingCompiler()

    def MultiThreadingCompiler(self):
        def Running(Pieces, Sender):
            out = {}
            for x in Pieces:
                P = Pieces[x].CompileParticles()
                out[x] = P
            Sender.send(out)
        
        batch_s = float(len(self.EventDictionary)) / float(12)
        Pipe = []
        Prz = []
        
        bundle = {}
        for ev in self.EventDictionary:
            bundle[ev] = self.EventDictionary[ev]

            if len(bundle) == batch_s:
                recv, send = multiprocessing.Pipe(False)
                P = multiprocessing.Process(target = Running, args = (bundle, send, ))
                Pipe.append(recv)
                Prz.append(P)
                bundle = {}
        
        if len(bundle) != 0:
            recv, send = multiprocessing.Pipe(False)
            P = multiprocessing.Process(target = Running, args = (bundle, send, ))
            Pipe.append(recv)
            Prz.append(P)
       
        al = Alerting(Prz, self.__Verbose)
        al.Notice("COMPILING PARTICLES")
        for i in Prz:
            i.start()
        
        for i, j in zip(Prz, Pipe):
            con = j.recv()
            i.join()
            for t in con:
                self.EventDictionary[t] = con[t]
            al.ProgressAlert()
        al.Notice("COMPILING COMPLETE")


class TruthCompiler(EventCompiler):
    def __init__(self, FileDir, Debug = False):

        tree = "nominal"
        init_c = "truth_top_initialState_child"
        c = "truth_top_child"
        tj = "truthjet"
        mt = "mu_true"
        et = "el_true"
        tt = "truth_top"
        
        EventProperties = ["met_met", "met_phi"]
    
        # Truth Branches
        jet_truth = [tj + "_pt", tj + "_eta", tj + "_phi", tj + "_e", tj + "_flavour"]
        top_truth = [tt+"_pt", tt+"_eta", tt+"_phi", tt+"_e", tt+"_charge", "top_FromRes"]
        top_truth_init_child = [init_c+"_pt", init_c+"_eta", init_c+"_phi", init_c+"_e", "top_initialState_child_pdgid"]
        top_truth_child = [c+"_pt", c+"_eta", c+"_phi", c+"_e", c + "_pdgid"]

        # Detector Measurements
        el = ["el_pt", "el_eta", "el_phi", "el_e", "el_charge", et+"_type", et+"_origin", et+"_isPrompt"]
        mu = ["mu_pt", "mu_eta", "mu_phi", "mu_e", "mu_charge", mt+"_type", mt+"_origin", mt+"_isPrompt"]
        jet = ["jet_pt", "jet_eta", "jet_phi", "jet_e"]
 
        self.__EventProperties = BranchVariable(FileDir, tree, EventProperties)
        self.EVNT = self.__EventProperties.EventObjectMap
        
        self.__T_Top = EventCompiler(FileDir, tree, top_truth, Verbose = False, Debug = Debug)
        self.__T_init_Children = EventCompiler(FileDir, tree, top_truth_init_child, Verbose = False, Debug = Debug)
        self.__T_Children = EventCompiler(FileDir, tree, top_truth_child, Verbose = False, Debug = Debug)
        self.__T_Jet = EventCompiler(FileDir, tree, jet_truth, Verbose = False, Debug = Debug)
        self.__D_Electron = EventCompiler(FileDir, tree, el, Verbose = False, Debug = Debug)
        self.__D_Muon = EventCompiler(FileDir, tree, mu, Verbose = False, Debug = Debug)
        self.__D_Jet = EventCompiler(FileDir, tree, jet, Verbose = False, Debug = Debug)

        self.__T_Top.GenerateEvents()
        self.__T_init_Children.GenerateEvents()
        self.__T_Children.GenerateEvents()
        self.__T_Jet.GenerateEvents()
        self.__D_Electron.GenerateEvents()
        self.__D_Muon.GenerateEvents()
        self.__D_Jet.GenerateEvents()

        self.T_TopD = self.__T_Top.EventDictionary
        self.T_init_ChildD = self.__T_init_Children.EventDictionary
        self.T_ChildD = self.__T_Children.EventDictionary
        self.T_JetD = self.__T_Jet.EventDictionary
        self.D_ElecD = self.__D_Electron.EventDictionary
        self.D_MuD = self.__D_Muon.EventDictionary
        self.D_JetD = self.__D_Jet.EventDictionary

        self.EventDictionary = {}
        self.__Verbose = True
        self.MultiThreading()

    def FindCommonIndex(self, incom, truth, init = True):
        for i in range(len(truth)):
            for j in range(len(incom)):
                if truth[i].Index == incom[j].Index:
                    truth[i].AddProduct(incom[j], init)
 
    def MatchToTruth(self, Runs):
        
        Output = {}
        for i in Runs:
            
            E = Event()
            
            self.FindCommonIndex(self.T_ChildD[i], self.T_TopD[i], True)
            self.FindCommonIndex(self.T_ChildD[i], self.T_TopD[i], False)

            init_C = self.T_init_ChildD[i]
            C = self.T_ChildD[i]
            
            jet_T = self.T_JetD[i]
            jet_D = self.D_JetD[i]
            el_D = self.D_ElecD[i]
            mu_D = self.D_MuD[i]
          
            # First Match the Turth Jets to the detector Particles
            ad = []
            ad += jet_T
            ad += el_D
            ad += mu_D
            
            self.MatchClosestParticles(jet_T, jet_D, False)
            self.MatchClosestParticles(init_C, ad, True)
            self.MatchClosestParticles(C, ad, False)
            
            E.TruthMatch += self.T_TopD[i]
            E.DetectorParticles += jet_D
            E.DetectorParticles += el_D
            E.DetectorParticles += mu_D
            E.EventNumber = i
            E.MET = self.__EventProperties.EventObjectMap[i]["met_met"]
            E.MET_Phi = self.__EventProperties.EventObjectMap[i]["met_phi"]

            Output[i] = E 
        return Output

    def MatchClosestParticles(self, Parent, Children, init = True, TH = 0.1):
        Pairs = {} 
        for i in Parent:
            for j in Children:
                dR = i.KinematicDifference(j)
                if dR < TH:
                    Pairs[dR] = [j, i]
        
        sort = sorted(Pairs.items())
        used = []
        for k in sort:
            ch = k[1][0]
            pa = k[1][1]

            if ch in used:
                continue
            else:
                used.append(ch)
                if init:
                    pa.init_DecayProducts.append(ch)
                else:
                    pa.DecayProducts.append(ch)
    

    def MultiThreading(self):
        def Running(Runs, Sender):
            Out = self.MatchToTruth(Runs)
            Sender.send(Out)

        batch_S = float(len(self.T_TopD)) / float(12)
        Pipe = []
        Prz = []

        Bundle = []
        for i in self.T_TopD:
            Bundle.append(i)

            if len(Bundle) == batch_S:
                recv, send = multiprocessing.Pipe(False)
                P = multiprocessing.Process(target = Running, args = (Bundle, send, ))
                Pipe.append(recv)
                Prz.append(P)
                Bundle = []

        if len(Bundle) != 0:
            recv, send = multiprocessing.Pipe(False)
            P = multiprocessing.Process(target = Running, args = (Bundle, send, ))
            Pipe.append(recv)
            Prz.append(P)

        al = Alerting(Prz, self.__Verbose)
        al.Notice("COMPILING EVENTS AND MATCHING TRUTH")
        for i in Prz:
            i.start()
        
        for i, j in zip(Prz, Pipe):
            con = j.recv()
            i.join()
            for t in con:
                self.EventDictionary[t] = con[t]
            al.ProgressAlert()
        al.Notice("COMPILING COMPLETE")


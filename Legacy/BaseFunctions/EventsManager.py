from BaseFunctions.VariableManager import *
from BaseFunctions.Alerting import ErrorAlert, Alerting, Debugging
from BaseFunctions.ParticlesManager import * 
import BaseFunctions
import multiprocessing
from BaseFunctions.Misc import Threading
from time import sleep

class Event:
    def __init__(self):
        self.TruthMatch = []
        self.DetectorParticles = []
        self.TruthJets = []
        self.TruthParticles_init = []
        self.TruthParticles = []
        self.Leptons = []
        self.RCJets = []
        self.Jets = []
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
        self.GenerateEvents()
    
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
        

        def Compiler(Pieces):
            out = {}
            for x in Pieces:
                P = self.EventDictionary[x].CompileParticles()
                out[x] = P
            return out
       
        x = Threading(verb = self.__Verbose)
        x.MultiThreading(self.EventDictionary, Compiler, self.EventDictionary)
        
class TruthCompiler(EventCompiler):
    def __init__(self, FileDir, Debug = False, Verbose = False):

        tree = "nominal"
        init_c = "truth_top_initialState_child"
        c = "truth_top_child"
        tj = "truthjet"
        mt = "mu_true"
        et = "el_true"
        tt = "truth_top"

        EventProperties = ["met_met", "met_phi"]
    
        # Truth Branches
        jet_truth = [tj + "_pt", tj + "_eta", tj + "_phi", tj + "_e", tj + "_flavour", tj + "_flavour_extended"]
        top_truth = [tt+"_pt", tt+"_eta", tt+"_phi", tt+"_e", tt+"_charge", "top_FromRes"]
        top_truth_init_child = [init_c+"_pt", init_c+"_eta", init_c+"_phi", init_c+"_e", "top_initialState_child_pdgid"]
        top_truth_child = [c+"_pt", c+"_eta", c+"_phi", c+"_e", c + "_pdgid"]

        # Detector Measurements
        el = ["el_pt", "el_eta", "el_phi", "el_e", "el_charge", et+"_type", et+"_origin", et+"_isPrompt", et + "_isChargeFl"]
        el += ["el_topoetcone20", "el_ptvarcone20", "el_CF", "el_d0sig", "el_delta_z0_sintheta"]
        
        mu = ["mu_pt", "mu_eta", "mu_phi", "mu_e", "mu_charge", mt+"_type", mt+"_origin", mt+"_isPrompt"]
        mu += ["mu_topoetcone20", "mu_ptvarcone30", "mu_d0sig", "mu_delta_z0_sintheta"] 
        
        jet = ["jet_pt", "jet_eta", "jet_phi", "jet_e", "jet_truthflav", "jet_truthPartonLabel", "jet_isTrueHS", "jet_truthflavExtended"]
        jet += ["jet_jvt", "jet_isbtagged_DL1_60", "jet_isbtagged_DL1_70", "jet_isbtagged_DL1_77", "jet_isbtagged_DL1_85", "jet_DL1", "jet_DL1_pb", "jet_DL1_pc", "jet_DL1_pu"]
        jet += ["jet_isbtagged_DL1r_60", "jet_isbtagged_DL1r_70", "jet_isbtagged_DL1r_77", "jet_isbtagged_DL1r_85", "jet_DL1r", "jet_DL1r_pb", "jet_DL1r_pc", "jet_DL1r_pu"]

        rc = ["rcjet_pt", "rcjet_eta", "rcjet_phi", "rcjet_e", "rcjet_d12", "rcjet_d23"]
        rcsub = ["rcjetsub_pt", "rcjetsub_eta", "rcjetsub_phi", "rcjetsub_e"]
 
        self.__EventProperties = BranchVariable(FileDir, tree, EventProperties)
        self.EVNT = self.__EventProperties.EventObjectMap
        
        self.__T_Top = EventCompiler(FileDir, tree, top_truth, Verbose = Verbose, Debug = Debug)
        self.__T_init_Children = EventCompiler(FileDir, tree, top_truth_init_child, Verbose = Verbose, Debug = Debug)
        self.__T_Children = EventCompiler(FileDir, tree, top_truth_child, Verbose = Verbose, Debug = Debug)
        self.__T_Jet = EventCompiler(FileDir, tree, jet_truth, Verbose = Verbose, Debug = Debug)
        self.__D_Electron = EventCompiler(FileDir, tree, el, Verbose = Verbose, Debug = Debug)
        self.__D_Muon = EventCompiler(FileDir, tree, mu, Verbose = Verbose, Debug = Debug)
        self.__D_Jet = EventCompiler(FileDir, tree, jet, Verbose = Verbose, Debug = Debug)
        self.__RC = EventCompiler(FileDir, tree, rc, Verbose = Verbose, Debug = Debug)
        self.__RC_Sub = EventCompiler(FileDir, tree, rcsub, Verbose = Verbose, Debug = Debug)

        self.T_TopD = self.__T_Top.EventDictionary
        self.T_init_ChildD = self.__T_init_Children.EventDictionary
        self.T_ChildD = self.__T_Children.EventDictionary
        self.T_JetD = self.__T_Jet.EventDictionary
        self.D_ElecD = self.__D_Electron.EventDictionary
        self.D_MuD = self.__D_Muon.EventDictionary
        self.D_JetD = self.__D_Jet.EventDictionary
        self.D_RC = self.__RC.EventDictionary
        self.D_RCSub = self.__RC_Sub.EventDictionary

        self.EventDictionary = {}
        self.__Verbose = True
        
        x = Threading(verb = self.__Verbose)
        x.MultiThreading(self.T_TopD, self.MatchToTruth, self.EventDictionary)
            
    def FindCommonIndex(self, incom, truth, init = True):
        for i in range(len(truth)):
            for j in range(len(incom)):
                if truth[i].Index == incom[j].Index:

                    if "RCJet" in truth[i].Type:
                        truth[i].Sub_Jets += [incom[j]]
                    else:
                        truth[i].AddProduct(incom[j], init)
 
    def MatchToTruth(self, Runs):
        
        Output = {}
        for i in Runs:
            
            E = Event()
            
            self.FindCommonIndex(self.T_init_ChildD[i], self.T_TopD[i], True)
            self.FindCommonIndex(self.T_ChildD[i], self.T_TopD[i], False)
            self.FindCommonIndex(self.D_RCSub[i], self.D_RC[i], False)

            init_C = self.T_init_ChildD[i]
            C = self.T_ChildD[i]
            
            jet_T = self.T_JetD[i]
            jet_D = self.D_JetD[i]
            el_D = self.D_ElecD[i]
            mu_D = self.D_MuD[i]
            rc_D = self.D_RC[i]
          
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

            E.TruthJets += jet_T
            E.TruthParticles_init += init_C
            E.TruthParticles += C
            E.Leptons += el_D
            E.Leptons += mu_D
            E.Jets += jet_D

            E.RCJets += rc_D

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
   


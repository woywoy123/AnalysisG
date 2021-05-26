import ROOT
import numpy as np
from particle import Particle as hep_particle
from BaseFunctions.IO import *


def ParticleVector(pt, eta, phi, en):

    v = ROOT.Math.PtEtaPhiEVector()
    v.SetCoordinates(pt, eta, phi, en)
    return v; 

def BulkParticleVector(pt, eta, phi, en):
    Output = []
    for i in range(len(pt)):
        v = ParticleVector(pt[i], eta[i], phi[i], en[i])
        Output.append(v)
    
    return np.array(Output)

def SumMultiParticleVector(pt, eta, phi, en):

    v = ROOT.Math.PtEtaPhiEVector()   
    for i in range(len(pt)):
        v = v + ParticleVector(pt[i], eta[i], phi[i], en[i])
    return v; 

def SumVectors(vector):
    v = ROOT.Math.PtEtaPhiEVector()
    for i in vector:
        v += i
    return v

class Particle:
    def __init__(self):
        self.Charge = ""
        self.PDGID = ""
        self.FourVector = ""
        self.DecayProducts = []
        self.IsSignal = ""
        self.Index = "" 
        
        if self.PDGID != "":
            x = hep_particle.from_pdgid(self.PDGID)
            self.Name = x.name
    
    def SetKinematics(self, E, Pt, Phi, Eta):
        self.E = float(E)
        self.Pt = float(Pt)
        self.Phi = float(Phi)
        self.Eta = float(Eta)
        
        self.CalculateFourVector()
        self.Mass = self.FourVector.mass() / 1000.

    def CalculateFourVector(self):
        self.FourVector = ParticleVector(self.Pt, self.Eta, self.Phi, self.E)
    
    def AddProduct(self, ParticleDaughter):
        self.DecayProducts.append(ParticleDaughter)

    def ReconstructFourVectorFromProducts(self):
        
        vectors = []
        for i in self.DecayProducts:
            vectors.append(i.FourVector)
        self.ReconstructedFourVector = SumVectors(vectors)
        self.FourVector = self.ReconstructedFourVector
        self.Mass = self.ReconstructedFourVector.mass() / 1000.

    def KinematicDifference(self, Particle):
        pass


class SignalSpectator:
    
    def __init__(self, ResMask, Tree, Branches, file_dir):
        self.Branches = Branches
        self.Search = Branches + ResMask
        self.Mask = ResMask[0]
        self.Tree = Tree
        self.file_dir = file_dir
        self.EventContainer = {}
        self.Verbose = True
        self.Interval = 10
        self.a = 0
        
        self.ReadArrays()
        self.SortBranchMap()
        self.EventLoop()

    def ReadArrays(self):

        print("INFO::Reading Branches and Trees")
        reader = FastReading(self.file_dir)
        reader.ReadBranchFromTree(self.Tree, self.Search)
        reader.ConvertBranchesToArray(core = len(self.Search))
        
        self.Internal = reader.ArrayBranches[self.Tree]
        self.Mask = reader.ArrayBranches[self.Tree][self.Mask]
        print("INFO::Finished Reading the Files")
        
    def SortBranchMap(self):
        for i in self.Branches:
            string = i.split("_")
            kin = string[len(string)-1]
            if "phi" == kin:
                self.Phi = self.Internal[i]
            if "eta" == kin:
                self.Eta = self.Internal[i]
            if "pt" == kin:
                self.Pt = self.Internal[i]
            if "e" == kin:
                self.E = self.Internal[i]
            if "pdgid" == kin:
                self.PDG = self.Internal[i]
  
    def CreateParticles(self):
        for i in range(len(self.it_e)):

            P = Particle()
            P.SetKinematics(self.it_e[i], self.it_pt[i], self.it_phi[i], self.it_eta[i])
            P.IsSignal = self.it_sig[i]
            P.PDGID = self.it_pdg[i]
            P.Index = self.index
            if self.index == -1:
                P.Index = i

            if self.it_sig[i] == 1:
                self.Map["Signal"].append(P)
            if self.it_sig[i] == 0:
                self.Map["Spectator"].append(P)
            self.Map["All"].append(P)

    def ProcessLoop(self):
        parent = False 
        self.Map = {}
        self.Map["All"] = []
        self.Map["Signal"] = []
        self.Map["Spectator"] = []
        for i in range(len(self.pl_mk)):
            
            if isinstance(self.pl_e[i], np.float32) == True:
                parent = True
                break
            
            self.it_e = self.pl_e[i]
            self.it_pt = self.pl_pt[i]
            self.it_phi = self.pl_phi[i]
            self.it_eta = self.pl_eta[i]
            self.it_sig = [self.pl_mk[i]]*len(self.pl_eta[i])
            self.it_pdg = self.pl_pdg[i]
            self.index = i
            self.CreateParticles()
       
        if parent:
            self.it_e = self.pl_e
            self.it_pt = self.pl_pt
            self.it_phi = self.pl_phi
            self.it_eta = self.pl_eta
            self.it_sig = self.pl_mk
            self.it_pdg = self.pl_pdg
            self.index = -1
            self.CreateParticles()

        else:
            self.ParticleParent()

    def EventLoop(self):
        print("INFO::Entering the EventLoop Function")
        self.EventContainer = []
        for i in range(len(self.Mask)):
            self.pl_mk = self.Mask[i]
            self.pl_phi = self.Phi[i]
            self.pl_eta = self.Eta[i]
            self.pl_pt = self.Pt[i]
            self.pl_e = self.E[i]
            self.pl_pdg = self.PDG[i]
            self.ProcessLoop()
            self.EventContainer.append(self.Map)
            
            self.NLoop = i
            self.ProgressAlert()


        print("INFO::Finished EventLoop")               
    
    def ParticleParent(self):
        
        s = {}
        for i in self.Map["All"]:
            try:
                s[i.Index].AddProduct(i)
                s[i.Index].IsSignal = i.IsSignal
            except KeyError:
                s[i.Index] = Particle()
                s[i.Index].AddProduct(i)
                s[i.Index].IsSignal = i.IsSignal
        
        for i in s:
            s[i].ReconstructFourVectorFromProducts()

        self.Map["FakeParents"] = s

    def ProgressAlert(self):

        if self.Verbose == True:
            
            per = round(float(self.NLoop) / float(len(self.Mask))* 100)
            if  per > self.a:
                print("INFO::Progress " + str(per) + "%")
                self.a = self.a + self.Interval
                 




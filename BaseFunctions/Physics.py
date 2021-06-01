import ROOT
import numpy as np
from particle import Particle as hep_P
from BaseFunctions.IO import *
import multiprocessing


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
        self.Name = "NAN" 
    
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
    
    def SetPDG(self, PDG = ""):
        if PDG != "":
            self.PDGID = PDG

        if self.PDGID != "":
            try:
                self.Name = hep_P.from_pdgid(self.PDGID)
            except NameError:
                self.Name = "NotFound"

class SignalSpectator:
    
    def __init__(self, ResMask, Tree, Branches, file_dir):
        self.Branches = Branches
        self.Search = Branches + ResMask
        self.Mask = ResMask[0]
        self.Tree = Tree
        self.file_dir = file_dir
        self.EventContainer = []
        self.Verbose = True
        self.Interval = 10
        self.a = 0
        self.NLoop = 0
        
        self.ReadArrays()
        self.SortBranchMap()
        self.EventLoop()

    def ReadArrays(self):

        print("INFO::Reading Branches and Trees")
        reader = FastReading(self.file_dir)
        reader.ReadBranchFromTree(self.Tree, self.Search)
        reader.ConvertBranchesToArray()
        
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
  
    def CreateParticles(self, e, pt, phi, eta, index, sig, pdg, Map):
        for i in range(len(e)):
            P = Particle()
            P.SetKinematics(e[i], pt[i], phi[i], eta[i])
            P.IsSignal = sig[i]
            P.PDGID = pdg[i]
            P.Index = index
            if index == -1:
                P.Index = i

            if sig[i] == 1:
                Map["Signal"].append(P)
            if sig[i] == 0:
                Map["Spectator"].append(P)
            Map["All"].append(P)
        
        return Map

    def ProcessLoop(self, e, pt, phi, eta, mk, pdg):
        parent = False 
        Map = {}
        Map["All"] = []
        Map["Signal"] = []
        Map["Spectator"] = []

        params = []
        pipes = []
        for i in range(len(mk)):
            
            if isinstance(e[i], np.float32) == True:
                parent = True
                break

            Map = self.CreateParticles(e[i], pt[i], phi[i], eta[i], i, [mk[i]]*len(eta[i]), pdg[i], Map)

        if parent:
            Map = self.CreateParticles(e, pt, phi, eta, -1, mk, pdg, Map)
        
        if parent == False:
            Map = self.ParticleParent(Map)
        return Map

    def ParticleParent(self, Map):
        
        s = {}
        for i in Map["All"]:
            try:
                s[i.Index].AddProduct(i)
                s[i.Index].IsSignal = i.IsSignal
            except KeyError:
                s[i.Index] = Particle()
                s[i.Index].AddProduct(i)
                s[i.Index].IsSignal = i.IsSignal
        
        for i in s:
            s[i].ReconstructFourVectorFromProducts()

        Map["FakeParents"] = s
        return Map

    def BulkRunning(self, inst, sender):
        
        EventContainer = []
        for i in inst:
            Map = self.ProcessLoop(i[0], i[1], i[2], i[3], i[4], i[5])
            EventContainer.append(Map)
        sender.send(EventContainer)


    def EventLoop(self):
        print("INFO::Entering the EventLoop Function")
            
        Params = []
        Pipe = []
        processes = []
        bundle_s = 4000
        for i in range(len(self.Mask)):
            inst = [self.E[i], self.Pt[i], self.Phi[i], self.Eta[i], self.Mask[i], self.PDG[i]]
            Params.append(inst)
            
            if len(Params) == bundle_s:
                recv, send = multiprocessing.Pipe(False)
                p = multiprocessing.Process(target = self.BulkRunning, args = (Params, send, ))
                processes.append(p)
                Pipe.append(recv)
                Params = []
        
        if len(Params) != 0:
            recv, send = multiprocessing.Pipe(False)
            p = multiprocessing.Process(target = self.BulkRunning, args = (Params, send, ))
            processes.append(p)
            Pipe.append(recv)
            Params = []
       
        for i in processes:
            i.start()

        for i, j in zip(processes, Pipe):
            con = j.recv()
            i.join() 
            for t in con:
                self.EventContainer.append(t)
            self.NLoop += len(con)
            self.ProgressAlert()

        print("INFO::Finished EventLoop")               
    
    def ProgressAlert(self):

        if self.Verbose == True:
            
            per = round(float(self.NLoop) / float(len(self.Mask))* 100)
            if  per > self.a:
                print("INFO::Progress " + str(per) + "%")
                self.a = self.a + self.Interval
                 




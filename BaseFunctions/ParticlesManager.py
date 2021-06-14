from particle import Particle as hep_P
from BaseFunctions.Physics import ParticleVector
from BaseFunctions.Physics import SumVectors
import numpy as np

class Particle:
    def __init__(self):
        self.Charge = ""
        self.PDGID = ""
        self.Flavour = ""
        self.FourVector = ""
        self.DecayProducts = []
        self.IsSignal = ""
        self.Index = "" 
        self.Name = "NAN"
        self.Type = ""

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ 
    
    def SetKinematics(self, E, Pt, Phi, Eta):
        self.E = float(E)
        self.Pt = float(Pt)
        self.Phi = float(Phi)
        self.Eta = float(Eta)
        
        self.CalculateFourVector()
        self.Mass = self.FourVector.mass / 1000.

    def CalculateFourVector(self):
        self.FourVector = ParticleVector(self.Pt, self.Eta, self.Phi, self.E)
    
    def AddProduct(self, ParticleDaughter):

        if isinstance(ParticleDaughter, list):
            self.DecayProducts += ParticleDaughter
        elif isinstance(ParticleDaughter, np.ndarray):
            self.DecayProducts += list(ParticleDaughter)
        else:
            self.DecayProducts.append(ParticleDaughter)

    def ReconstructFourVectorFromProducts(self):
        vectors = []
        for i in self.DecayProducts:
            vectors.append(i.FourVector)

        self.ReconstructedFourVector = SumVectors(vectors)
        self.FourVector = self.ReconstructedFourVector
        self.Mass = self.ReconstructedFourVector.mass / 1000.

    def KinematicDifference(self, Particle):
        DeltaPhi = self.Phi - Particle.Phi
        DeltaEta = self.Eta - Particle.Eta
        DeltaR = pow( pow(DeltaPhi, 2) + pow(DeltaEta, 2) , 0.5)
        return DeltaR 

    def PurgeChildrendR(self, dR):
        update = []
        for i in self.DecayProducts:
            if dR > self.KinematicDifference(i):
                update.append(i)
        self.DecayProducts = update

    def SetPDG(self, PDG = ""):
        if PDG != "":
            self.PDGID = PDG

        if self.PDGID != "":
            try:
                self.Name = hep_P.from_pdgid(self.PDGID)
            except NameError:
                self.Name = "NotFound"


class Lepton(Particle):
    def __init__(self):
        super().__init__(self)
        
        # Detector Measurements
        self.topoetcone20 = ""
        self.ptvarcone20 = ""
        self.CF = ""
        self.d0sig = ""
        self.deltaz0sintheta = ""

        # Monte Carlo Truth 
        self.true_type = ""
        self.true_origin = ""
        self.true_firstEgMotherTruthType = ""
        self.true_firstEgMotherTruthOrigin = ""
        self.true_firstEgMotherPdgId = ""
        self.true_IFFclass = ""
        self.true_isPrompt = ""
        self.true_isChargeFl = ""


class CreateParticleObjects:
    def __init__(self, E, Pt, Phi, Eta):
       
        self.Charge = ""
        self.PDGID = ""
        self.Flavour = ""
        self.Mask = ""
        self.E = E
        self.Pt = Pt
        self.Phi = Phi
        self.Eta = Eta
        self.Type = ""

    def CompileParticles(self):
        Output = [] 
        for i in range(len(self.E)):
            try:
                float(self.E[i])
                P = self.SingleParticles(i)
            except TypeError:
                P = self.MultiParticles(i)
            Output += P
        return Output

    def MultiParticles(self, i):
        Output = []
        for j in range(len(self.E[i])):
            P = Particle()
            P.SetKinematics(self.E[i][j], self.Pt[i][j], self.Phi[i][j], self.Eta[i][j])
            P.Index = i
            P.Type = self.Type
    
            if isinstance(self.Charge, str) == False:
                P.Charge = self.Charge[i][j]
            if isinstance(self.PDGID, str) == False:
                P.PDGID = self.PDGID[i][j]
            if isinstance(self.Mask, str) == False:
                P.IsSignal = self.Mask[i]
            if isinstance(self.Flavour, str) == False:
                P.Flavour = self.Flavour[i][j]
            Output.append(P)
        return Output

    def SingleParticles(self, i):
        Output = []
        P = Particle()
        P.SetKinematics(self.E[i], self.Pt[i], self.Phi[i], self.Eta[i])
        P.Index = i
        P.Type = self.Type
        
        if isinstance(self.Charge, str) == False:
            P.Charge = self.Charge[i]
        if isinstance(self.PDGID, str) == False:
            P.PDGID = self.PDGID[i]
        if isinstance(self.Mask, str) == False:
            P.IsSignal = self.Mask[i]
        if isinstance(self.Flavour, str) == False:
            P.Flavour = self.Flavour[i]
        
        Output.append(P)
        return Output







from BaseFunctions.Physics import ParticleVector, SumVectors
from BaseFunctions.VariableManager import VariableObjectProxy
import numpy as np

class Particle:
    def __init__(self):
        self.Charge = ""
        self.PDGID = ""
        self.Flavour = ""
        self.FourVector = ""
        self.DecayProducts = []
        self.init_DecayProducts = []
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
        
        self.Theta = self.FourVector.theta()
        self.Mass = self.FourVector.mass / 1000.

    def CalculateFourVector(self):
        self.FourVector = ParticleVector(self.Pt, self.Eta, self.Phi, self.E)
    
    def AddProduct(self, ParticleDaughter, init = True):
        
        if init == True:
            if isinstance(ParticleDaughter, list):
                self.init_DecayProducts += ParticleDaughter
            elif isinstance(ParticleDaughter, np.ndarray):
                self.init_DecayProducts += list(ParticleDaughter)
            else:
                self.init_DecayProducts.append(ParticleDaughter)
        
        if init == False:
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

class Lepton(Particle):
    def __init__(self):
        Particle.__init__(self)
        
        # Detector Measurements
        self.topoetcone20 = ""
        self.ptvarcone20 = ""
        self.ptvarcone30 = ""
        self.CF = ""
        self.d0sig = ""
        self.delta_z0_sintheta = ""

        # Monte Carlo Truth 
        self.true_type = ""
        self.true_origin = ""
        self.true_firstEgMotherTruthType = ""
        self.true_firstEgMotherTruthOrigin = ""
        self.true_firstEgMotherPdgId = ""
        self.true_IFFclass = ""
        self.true_isPrompt = ""
        self.true_isChargeFl = ""

class Jet(Particle):
    def __init__(self):
        Particle.__init__(self)
        self.nChad = ""
        self.nBhad = ""
        self.truthflav = ""
        self.truthPartonLabel = ""
        self.isTrueHS = ""
        self.jvt = ""

        self.DL1r = ""
        self.DL1r_60 = ""
        self.DL1r_70 = ""
        self.DL1r_77 = ""
        self.DL1r_85 = ""
        self.DL1r_pb = ""
        self.DL1r_pc = ""
        self.DL1r_pu = ""

        self.DL1 = ""
        self.DL1_60 = ""
        self.DL1_70 = ""
        self.DL1_77 = ""
        self.DL1_85 = ""
        self.DL1_pb = ""
        self.DL1_pc = ""
        self.DL1_pu = ""

        self.Sub_Jets = []

class CreateParticleObjects(Lepton, Jet, Particle):
    def __init__(self, E, Pt, Phi, Eta, Type = ""):
        if Type == "Muon" or Type == "Electron":
            Lepton.__init__(self)
        elif Type == "Jet" or Type == "RCJet" or Type == "TruthJet":
            Jet.__init__(self)
        else:
            Particle.__init__(self)
        self.E = E
        self.Pt = Pt
        self.Phi = Phi
        self.Eta = Eta
        self.Type = Type

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
            if self.Type == "Muon" or self.Type == "Electron":
                P = Lepton()
            elif self.Type == "Jet" or self.Type == "RCJet" or self.Type == "TruthJet":
                P = Jet()
            else:
                P = Particle()
            P.SetKinematics(self.E[i][j], self.Pt[i][j], self.Phi[i][j], self.Eta[i][j])
            P.Index = i
            P.Type = self.Type

            if isinstance(self.IsSignal, str) == False:
                P.IsSignal = self.IsSignal[i]
            
            VariableObjectProxy(self, P, i, j)
            Output.append(P)
        return Output

    def SingleParticles(self, i):
        Output = []
        if self.Type == "Muon" or self.Type == "Electron":
            P = Lepton()
        elif self.Type == "Jet" or self.Type == "RCJet" or self.Type == "TruthJet":
            P = Jet()
        else:
            P = Particle()
        P.SetKinematics(self.E[i], self.Pt[i], self.Phi[i], self.Eta[i])
        P.Index = i
        P.Type = self.Type

        VariableObjectProxy(self, P, i)
        Output.append(P)
        return Output







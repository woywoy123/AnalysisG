
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
    
    def SetKinematics(self, E, Pt, Phi, Eta):
        self.E = float(E)
        self.Pt = float(Pt)
        self.Phi = float(Phi)
        self.Eta = float(Eta)
        
        self.CalculateFourVector()
        self.Mass = self.FourVector.mass() / 1000.

    def CalculateFourVector(self):
        from BaseFunctions.Physics import ParticleVector
        self.FourVector = ParticleVector(self.Pt, self.Eta, self.Phi, self.E)
    
    def AddProduct(self, ParticleDaughter):
        self.DecayProducts.append(ParticleDaughter)

    def ReconstructFourVectorFromProducts(self):
        
        vectors = []
        for i in self.DecayProducts:
            vectors.append(i.FourVector)

        from BaseFunctions.Physics import SumVectors
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

def CreateParticles(e, pt, phi, eta, pdg = [], index = "", sig = [], flavour = []):
    Output = [] 

    for i in range(len(e)):
        P = Particle()

        P.SetKinematics(e[i], pt[i], phi[i], eta[i])
        if len(sig) == len(e):
            P.IsSignal = sig[i]

        if len(pdg) == len(e):
            P.PDGID = pdg[i]

        if len(flavour) == len(e):
            P.Flavour = flavour[i]


        if isinstance(index, str) == False:
            P.Index = index
            if index == -1:
                P.Index = i

        Output.append(P)

    return Output



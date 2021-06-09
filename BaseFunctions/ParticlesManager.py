
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
        from BaseFunctions.Physics import ParticleVector
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

def CreateParticles(self, e = [], pt = [], phi = [], eta = [], pdg = [], index = "", sig = []):
    Output = [] 
    for i in range(len(e)):
        P = Particle()
        P.SetKinematics(e[i], pt[i], phi[i], eta[i])
        if len(sig) == len(e):
            P.IsSignal = sig[i]

        if len(pdg) == len(e):
            P.PDGID = pdg[i]
        
        if isinstance(index, str) == False:
            P.Index = index
            if index == -1:
                P.Index = i

        Output.append(P)

    return Output



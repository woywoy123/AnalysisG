from AnalysisTopGNN.Templates import ParticleTemplate

# Particle Definitions 
class Particle(ParticleTemplate):
    def __init__(self):
        ParticleTemplate.__init__(self)
        self.Type = "Particle"
        self.PID = self.Type + ".PID"
        self.Status = self.Type + ".Status"
        self.Charge = self.Type + ".Charge"
        self.PT = self.Type + ".PT"
        self.Eta = self.Type + ".Eta"
        self.Phi = self.Type + ".Phi"
        self.E = self.Type + ".E"
        self.Mass = self.Type + ".Mass"
        self.Daughter1 = self.Type + ".D1"
        self.Daughter2 = self.Type + ".D2"
        self.Mother1 = self.Type + ".M1"
        self.Mother2 = self.Type + ".M2"
        self._DefineParticle()
        self.Index = 0
        self.Decay_init = []
        self.Decay = []
        
class Jet(ParticleTemplate):
    def __init__(self):
        ParticleTemplate.__init__(self)
        self.Type = "Jet"
        self.Charge = self.Type + ".Charge"
        self.PT = self.Type + ".PT"
        self.Eta = self.Type + ".Eta"
        self.Phi = self.Type + ".Phi"
        self.Mass = self.Type + ".Mass"
        self.BTag = self.Type + ".BTag"
        self.Reference = self.Type + ".Particles"
        self.Constituents = self.Type + ".Constituents"
        self._DefineParticle() 

class GenJet(ParticleTemplate):
    def __init__(self):
        ParticleTemplate.__init__(self)
        self.Type = "GenJet"
        self.Charge = self.Type + ".Charge"
        self.PT = self.Type + ".PT"
        self.Eta = self.Type + ".Eta"
        self.Phi = self.Type + ".Phi"
        self.Mass = self.Type + ".Mass"
        self.Reference = self.Type + ".Particles"
        self.Constituents = self.Type + ".Constituents"
        self._DefineParticle() 

class Electron(ParticleTemplate):
    def __init__(self):
        ParticleTemplate.__init__(self)
        self.Type = "Electron"
        self.Charge = self.Type + ".Charge"
        self.PT = self.Type + ".PT"
        self.Eta = self.Type + ".Eta"
        self.Phi = self.Type + ".Phi"
        self.Reference = self.Type + ".Particle"
        self._DefineParticle() 

class Muon(ParticleTemplate):
    def __init__(self):
        ParticleTemplate.__init__(self)
        self.Type = "Muon"
        self.Charge = self.Type + ".Charge"
        self.PT = self.Type + ".PT"
        self.Eta = self.Type + ".Eta"
        self.Phi = self.Type + ".Phi"
        self.Reference = self.Type + ".Particle"
        self._DefineParticle()  

class Photon(ParticleTemplate):
    def __init__(self):
        ParticleTemplate.__init__(self)
        self.Type = "Photon"
        self.PT = self.Type + ".PT"
        self.Eta = self.Type + ".Eta"
        self.Phi = self.Type + ".Phi"
        self.E = self.Type + ".E"
        self.Reference = self.Type + ".Particles"
        self._DefineParticle()

class MissingET(ParticleTemplate):
    def __init__(self):
        ParticleTemplate.__init__(self)
        self.Type = "MissingET"
        self.MET = self.Type + ".MET"
        self.Eta = self.Type + ".Eta"
        self.Phi = self.Type + ".Phi"
        self._DefineParticle()



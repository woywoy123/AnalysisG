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
        
        self._DefineParticle() 

class Jet(ParticleTemplate):
    def __init__(self):
        ParticleTemplate.__init__(self)
        self.Type = "Jet"
        self.Charge = self.Type + ".Charge"
        
        self._DefineParticle()
        


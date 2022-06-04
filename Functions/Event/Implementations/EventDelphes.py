from Functions.Event.EventTemplate import EventTemplate 
from Functions.Particles.ParticleTemplate import Particle as Part

# Particle Definitions 
class Particle(Part):
    def __init__(self):
        Part.__init__(self)
        self.Type = "Particle"
        self.PID = self.Type + ".PID"
        self.Status = self.Type + ".Status"
        self.Charge = self.Type + ".Charge"
        self.PT = self.Type + ".PT"
        self.Eta = self.Type + ".Eta"
        self.Phi = self.Type + ".Phi"
        self.E = self.Type + ".E"
        self.__DefineParticle() 


class Event(EventTemplate):
    def __init__(self):
        EventTemplate.__init__(self)
        self.runNumber = "Event.Number"
        self.Weight = "Event.Weight"
        self.Tree = ["Delphes"]
        self.Branches = ["Particle"]
        self.Objects = {
                "Particle" : Particle()
                        }
        self.DefineObjects()

    def CompileEvent(self, ClearVal):
        self.CompileParticles(False)
        self.Particle = self.DictToList(self.Particle)


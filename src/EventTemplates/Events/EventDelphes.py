from AnalysisTopGNN.Templates import EventTemplate 
from AnalysisTopGNN.Particles.DelphesParticles import Particle

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
        self.CompileParticles(ClearVal)
        self.Particle = self.DictToList(self.Particle)


from AnalysisTopGNN.Templates import EventTemplate 
from AnalysisTopGNN.Particles.DelphesParticles import Particle, Jet

class Event(EventTemplate):
    def __init__(self):
        EventTemplate.__init__(self)
        self.runNumber = "Event.Number"
        self.Weight = "Event.Weight"
        self.Tree = ["Delphes"]
        self.Branches = ["Particle", "Jet"]
        self.Objects = {
                "Particle" : Particle(), 
                "Jets" : Jet(),
                        }
        self.DefineObjects()

    def CompileEvent(self, ClearVal):
        self.CompileParticles(ClearVal)
        self.Particle = self.DictToList(self.Particle)
        print(self.Jets)


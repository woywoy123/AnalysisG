from AnalysisG.Templates import GraphTemplate

class GraphTops(GraphTemplate):
    def __init__(self, Event = None):
        GraphTemplate.__init__(self)
        self.Event = Event
        self.Particles += self.Event.Tops

class GraphChildren(GraphTemplate):
    def __init__(self, Event = None):
        GraphTemplate.__init__(self)
        self.Event = Event
        self.Particles += self.Event.TopChildren

class GraphChildrenNoNu(GraphTemplate):
    def __init__(self, Event = None):
        GraphTemplate.__init__(self)
        self.Event = Event
        self.Particles += [i for i in self.Event.TopChildren if abs(i.pdgid) not in [12, 14, 16]]

class GraphTruthJetLepton(GraphTemplate):
    def __init__(self, Event = None):
        GraphTemplate.__init__(self)
        self.Event = Event
        self.Particles += self.Event.TruthJets
        self.Particles += self.Event.Electrons
        self.Particles += self.Event.Muons

class GraphDetector(GraphTemplate):
    def __init__(self, Event = None):
        GraphTemplate.__init__(self)
        self.Event = Event
        self.Particles += self.Event.DetectorParticles



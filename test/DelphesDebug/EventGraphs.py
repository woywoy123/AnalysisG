from AnalysisTopGNN.Templates import EventGraphTemplate

class EventGraphTruthTops(EventGraphTemplate):
    def __init__(self, Event):
        EventGraphTemplate.__init__(self)
        self.Event = Event
        self.Particles += self.Event.TruthTops

class EventGraphTruthTopChildren(EventGraphTemplate):
    def __init__(self, Event):
        EventGraphTemplate.__init__(self)
        self.Event = Event
        self.Particles += self.Event.Final_Particles
    
class EventGraphTruthJetLepton(EventGraphTemplate):
    def __init__(self, Event):
        EventGraphTemplate.__init__(self)
        self.Event = Event
        self.Particles += self.Event.TruthJets
        self.Particles += self.Event.Electrons
        self.Particles += self.Event.Muons

class EventGraphDetector(EventGraphTemplate):
    def __init__(self, Event):
        EventGraphTemplate.__init__(self)
        self.Event = Event
        self.Particles += self.Event.DetectorParticles



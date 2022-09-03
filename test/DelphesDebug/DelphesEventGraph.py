from AnalysisTopGNN.Templates import EventGraphTemplate
class EventGraphTruthTopChildren(EventGraphTemplate):
    def __init__(self, Event):
        EventGraphTemplate.__init__(self)
        self.Event = Event
        self.Particles += Event.Final_Particles


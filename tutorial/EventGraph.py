from AnalysisTopGNN.Templates import EventGraphTemplate

class SimpleDataGraph(EventGraphTemplate):

    def __init__(self, Event = None):
        EventGraphTemplate.__init__(self)
        self.Event = Event 
        self.Particles += Event.Tops

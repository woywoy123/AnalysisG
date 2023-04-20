from AnalysisG.Templates import GraphTemplate

class SimpleDataGraph(GraphTemplate):

    def __init__(self, Event = None):
        GraphTemplate.__init__(self)
        self.Event = Event 
        self.Particles += Event.Tops

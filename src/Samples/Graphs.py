from AnalysisTopGNN.Features import FeatureAnalysis
from AnalysisTopGNN.Tools import Tools
from AnalysisTopGNN.Generators import Settings

class Graphs:
    
    def __init__(self):
        pass #Settings.__init__(self)


    def GetEventGraph(self):
        name = self.EventGraph.__init__.__qualname__.split(".")[-2]
        _, evnt = self.GetObjectFromString(self.EventGraph.__module__, name)
        return evnt

    def MakeGraph(self, event, smplidx):
        ev = self.CopyInstance(self.EventGraph)
        try:
            ev = ev(event)
        except AttributeError:
            ev = ev.Escape(ev)
            ev.Event = event
            ev.Particles = []

        ev.iter = smplidx
        ev.SelfLoop = self.SelfLoop
        ev.FullyConnect = self.FullyConnect
        ev.EdgeAttr |= self.EdgeAttribute
        ev.NodeAttr |= self.NodeAttribute
        ev.GraphAttr |= self.GraphAttribute
        return ev





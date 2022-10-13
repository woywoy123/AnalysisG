from AnalysisTopGNN.Features import FeatureAnalysis
from AnalysisTopGNN.Tools import Tools

class Graphs(FeatureAnalysis, Tools):

    def __init__(self):
        self.Device = True
        self.SelfLoop = True
        self.FullyConnect = True
        self.GraphAttribute = {}
        self.NodeAttribute = {}
        self.EdgeAttribute = {}
        self.EventGraph = None
        self.Tree = False

    def SetAttribute(self, c_name, fx, container):
        if c_name == "P_" or c_name == "T_":
            c_name += fx.__name__ 
        elif c_name == "":
            c_name += fx.__name__ 

        if c_name not in container:
            container[c_name] = fx
        else:
            self.Warning("Found Duplicate " + c_name + " Attribute")

    def GetEventGraph(self):
        name = self.EventGraph.__init__.__qualname__.split(".")[-2]
        _, evnt = self.GetObjectFromString(self.EventGraph.__module__, name)
        return evnt

    def MakeGraph(self, event, smplidx):
        ev = self.GetEventGraph()
        ev = ev(event)
        ev.iter = smplidx
        ev.SelfLoop = self.SelfLoop
        ev.FullyConnect = self.FullyConnect
        ev.EdgeAttr |= self.EdgeAttribute
        ev.NodeAttr |= self.NodeAttribute
        ev.GraphAttr |= self.GraphAttribute
        return ev





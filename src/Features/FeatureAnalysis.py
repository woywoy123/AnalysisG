import sys
from AnalysisTopGNN.Notification import FeatureAnalysis

class FeatureAnalysis(FeatureAnalysis):
    def __init__(self):
        pass

    def TestGraphFeature(self, Event, EventGraph, Fx):
        return Fx(EventGraph(Event).Event)

    def TestNodeFeature(self, Event, EventGraph, Fx):
        return [ Fx(i) for i in EventGraph(Event).Particles]

    def TestEdgeFeature(self, Event, EventGraph, Fx):
        ev = EventGraph(Event)
        return [ Fx(i, j) for i in ev.Particles for j in ev.Particles]

    def TestEvent(self, Event, EventGraph):
        if isinstance(Event, list):
            for ev in Event:
                self.TestEvent(ev, EventGraph)
            return 
        if hasattr(Event, "Trees"):
            for tree in Event.Trees:
                self.TestEvent(Event.Trees[tree], EventGraph)
            return 
        
        res = {"GRAPH" : {}, "NODE" : {}, "EDGE" : {}}
        for c_name in self.GraphAttribute:
            try:
                self.TestGraphFeature(Event, self.GetEventGraph(), self.GraphAttribute[c_name])    
            except AttributeError:
                fail = str(sys.exc_info()[1]).replace("'", "").split(" ")
                res["GRAPH"][c_name] = "FAILED: " + Fx_m + " -> " + Fx_n + " ERROR -> " + " ".join(fail)

        for c_name in self.NodeAttribute:
            try:
                self.TestNodeFeature(Event, self.GetEventGraph(), self.NodeAttribute[c_name])    
            except AttributeError:
                fail = str(sys.exc_info()[1]).replace("'", "").split(" ")
                res["NODE"][c_name] = "FAILED: " + Fx_m + " -> " + Fx_n + " ERROR -> " + " ".join(fail)

        for c_name in self.EdgeAttribute:
            try:
                self.TestEdgeFeature(Event, self.GetEventGraph(), self.EdgeAttribute[c_name])    
            except AttributeError:
                fail = str(sys.exc_info()[1]).replace("'", "").split(" ")
                res["EDGE"][c_name] = "FAILED: " + Fx_m + " -> " + Fx_n + " ERROR -> " + " ".join(fail)

        #// Continue here with the error notification ...
